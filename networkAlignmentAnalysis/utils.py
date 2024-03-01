from warnings import warn
import zipfile
import os
from typing import List
from contextlib import contextmanager
from functools import wraps
import numpy as np
from scipy.linalg import null_space
from sklearn.decomposition import IncrementalPCA
import torch
from gitignore_parser import parse_gitignore


# -------------- context managers & decorators --------------
@contextmanager
def no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield


def test_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to eval
        in_training_mode = [set_net_mode(net, training=False) for net in nets]

        # do decorated function
        func_outputs = func(nets, *args, **kwargs)

        # return networks to whatever mode they used to be in
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)

        # return decorated function outputs
        return func_outputs

    # return decorated function
    return wrapper


def train_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to train
        in_training_mode = [set_net_mode(net, training=True) for net in nets]

        # do decorated function
        func_outputs = func(nets, *args, **kwargs)

        # return networks to whatever mode they used to be in
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)

        # return decorated function outputs
        return func_outputs

    # return decorated function
    return wrapper


def set_net_mode(net, training=True):
    """helper for setting mode of network and returning current mode"""
    # get current mode of network
    in_training_mode = net.training
    # set to training mode or evaluation mode
    if training:
        net.train()
    else:
        net.eval()
    # return original mode of network
    return in_training_mode


# ------- some other things -------
def get_device(obj):
    """simple method to get device of input tensor or nn.Module"""
    if isinstance(obj, torch.nn.Module):
        return next(obj.parameters()).device.type
    elif isinstance(obj, torch.Tensor):
        return "cuda" if obj.is_cuda else "cpu"
    else:
        raise ValueError("")


def check_iterable(val):
    """duck-type check if val is iterable"""
    try:
        _ = iter(val)
    except:
        return False
    else:
        return True


def remove_by_idx(input, idx, dim):
    """
    remove part of input indexed by idx on dim
    """
    idx_keep = [i for i in range(input.size(dim)) if i not in idx]
    return torch.index_select(input, dim, torch.tensor(idx_keep).to(input.device))


def get_eval_transform_by_cutoff(cutoff):
    """
    get method for transforming eigenvalues into a binary keep fraction

    will scale each eigenvector by 1 or 0 depending on whether that eigenvalue
    explains more than **cutoff** fraction of the variance

    returns a callable method
    """

    def eval_transform(evals):
        assert torch.all(
            evals >= 0
        ), "found negative eigenvalues, doesn't work for 'cutoff' eval_transform"
        evals = evals / torch.sum(evals)
        return 1.0 * (evals > cutoff)

    return eval_transform


def smartcorr(input):
    """
    Performs torch corrcoef on the input data but sets each pair-wise correlation coefficent
    to 0 where the activity has no variance (var=0) for a particular dimension (replaces nans with zeros)sss
    """
    idx_zeros = torch.var(input, dim=1) == 0
    cc = torch.corrcoef(input)
    cc[idx_zeros, :] = 0
    cc[:, idx_zeros] = 0
    return cc


def batch_cov(input, centered=True, correction=True):
    """
    Performs batched covariance on input data of shape (batch, dim, samples) or (dim, samples)

    Where the resulting batch covariance matrix has shape (batch, dim, dim) or (dim, dim)
    and bcov[i] = torch.cov(input[i]) if input.ndim==3

    if centered=True (default) will subtract the means first

    if correction=True, will use */(N-1) otherwise will use */N
    """
    assert (input.ndim == 2) or (input.ndim == 3), "input must be a 2D or 3D tensor"
    assert isinstance(correction, bool), "correction must be a boolean variable"

    # check if batch dimension was provided
    no_batch = input.ndim == 2

    # add an empty batch dimension if not provided
    if no_batch:
        input = input.unsqueeze(0)

    # measure number of samples of each input matrix
    S = input.size(2)

    # subtract mean if doing centered covariance
    if centered:
        input = input - input.mean(dim=2, keepdim=True)

    # measure covariance of each input matrix
    bcov = torch.bmm(input, input.transpose(1, 2))

    # correct for number of samples
    bcov /= S - 1.0 * correction

    # remove empty batch dimension if not provided
    if no_batch:
        bcov = bcov.squeeze(0)

    return bcov


def smart_pca(input, centered=True, use_rank=True, correction=True):
    """
    smart algorithm for pca optimized for speed

    input should either have shape (batch, dim, samples) or (dim, samples)
    if dim > samples, will use svd and if samples < dim will use covariance/eigh method

    will center data when centered=True

    if it fails, will fall back on performing sklearns IncrementalPCA whenever forcetry=True
    """
    assert (input.ndim == 2) or (input.ndim == 3), "input should be a matrix or batched matrices"
    assert isinstance(correction, bool), "correction should be a boolean"

    if input.ndim == 2:
        no_batch = True
        input = input.unsqueeze(0)  # create batch dimension for uniform code
    else:
        no_batch = False

    _, D, S = input.size()
    if D > S:
        # if more dimensions than samples, it's more efficient to run svd
        v, w, _ = named_transpose([torch.linalg.svd(inp) for inp in input])
        # convert singular values to eigenvalues
        w = [ww**2 / (S - 1.0 * correction) for ww in w]
        # append zeros because svd returns w in R**k where k = min(D, S)
        w = [torch.concatenate((ww, torch.zeros(D - S))) for ww in w]

    else:
        # if more samples than dimensions, it's more efficient to run eigh
        bcov = batch_cov(input, centered=centered, correction=correction)
        w, v = named_transpose([eigendecomposition(C, use_rank=use_rank) for C in bcov])

    # return to stacked tensor across batch dimension
    w = torch.stack(w)
    v = torch.stack(v)

    # if no batch originally provided, squeeze out batch dimension
    if no_batch:
        w = w.squeeze(0)
        v = v.squeeze(0)

    # return eigenvalues and eigenvectors
    return w, v


def eigendecomposition(C, use_rank=True):
    """
    helper for getting eigenvalues and eigenvectors of covariance matrix

    will measure eigenvalues and eigenvectors with torch.linalg.eigh()
    the output will be sorted from highest to lowest eigenvalue (& eigenvector)

    if use_rank=True, will measure the rank of the covariance matrix and zero
    out any eigenvalues beyond the rank (that are usually nonzero numerical errors)
    """
    try:
        # measure eigenvalues and eigenvectors
        w, v = torch.linalg.eigh(C)

    except torch._C._LinAlgError as error:
        # this happens if the algorithm failed to converge
        # try with sklearn's incrementalPCA algorithm
        return sklearn_pca(C, use_rank=use_rank)

    except Exception as error:
        # if any other exception, raise it
        raise error

    # sort by eigenvalue from highest to lowest
    w_idx = torch.argsort(-w)
    w = w[w_idx]
    v = v[:, w_idx]

    # iff use_rank=True, will set eigenvalues to 0 for probable numerical errors
    if use_rank:
        crank = torch.linalg.matrix_rank(C)  # measure rank of covariance
        w[crank:] = 0  # set eigenvalues beyond rank to 0

    # return eigenvalues and eigenvectors
    return w, v


def sklearn_pca(input, use_rank=True, rank=None):
    """
    sklearn incrementalPCA algorithm serving as a replacement for eigh when it fails

    input should be a tensor with shape (num_samples, num_features) or it can be a
    covariance matrix with (num_features, num_features)

    if use_rank=True, will set num_components to the rank of input and then fill out the
    rest of the components with random orthogonal components in the null space of the true
    components and set the eigenvalues to 0

    if use_rank=False, will attempt to fit all the components
    if rank is not None, will attempt to fit #=rank components without measuring the rank directly
    (will ignore "rank" if use_rank=False)

    returns w, v where w is eigenvalues and v is eigenvectors sorted from highest to lowest
    """
    # dimension
    num_samples, num_features = input.shape

    # measure rank (or set to None)
    rank = None if not use_rank else (rank if rank is not None else fast_rank(input))

    # create and fit IncrementalPCA object on input data
    ipca = IncrementalPCA(n_components=rank).fit(input)

    # eigenvectors are the components
    v = ipca.components_

    # eigenvalues are the scaled singular values
    w = ipca.singular_values_**2 / num_samples

    # if v is a subspace of input (e.g. not a full basis, fill it out)
    if v.shape[0] < num_features:
        msg = "adding this because I think it should always be true, and if not I want to find out"
        assert w.shape[0] == v.shape[0], msg
        v_kernel = null_space(v).T
        v = np.vstack((v, v_kernel))
        w = np.concatenate((w, np.zeros(v_kernel.shape[0])))

    return torch.tensor(w, dtype=torch.float), torch.tensor(v, dtype=torch.float).T


def fast_rank(input):
    """uses transpose to speed up rank computation, otherwise normal"""
    if input.size(-2) < input.size(-1):
        input = torch.transpose(input, -2, -1)
    return int(torch.linalg.matrix_rank(input))


# ------------------ alignment functions ----------------------
def alignment(input, weight, method="alignment"):
    """
    measure alignment (proportion variance explained) between **input** and **weight**

    computes the rayleigh quotient between each weight vector in **weight** and the **input** fed
    into **weight**. Typically, **input** is the output in Layer L-1 and **weight** is from Layer L

    the output is normalized by the total variance in output of layer L-1 to measure the proportion
    of variance of in **input** is explained by a projection onto node's weights in **weight**

    args
    ----
        input: (batch, neurons) torch tensor
            - represents input activity being fed in to network weight layer
        weight: (num_out, num_in) torch tensor
            - represents weights multiplied by input layer
        method: string, default='alignment'
            - which method to use to measure structure in **input**
            - if 'alignment', uses covariance matrix of **input**
            - if 'similarity', uses correlation matrix of **input**

    returns
    -------
        alignment: (num_out, ) torch tensor
            - proportion of variance explained by projection of **input** onto each **weight** vector
    """
    assert (
        method == "alignment" or method == "similarity"
    ), "method must be set to either 'alignment' or 'similarity' (or None, default is alignment)"
    if method == "alignment":
        cc = torch.cov(input.T)
    elif method == "similarity":
        cc = smartcorr(input.T)
    else:
        raise ValueError(
            f"did not recognize method ({method}), must be 'alignment' or 'similarity'"
        )
    # Compute rayleigh quotient
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)
    # proportion of variance explained by a projection of the input onto each weight
    return rq / torch.trace(cc)


def get_maximum_strides(h_input, w_input, layer):
    h_max = int(
        np.floor(
            (h_input + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1)
            / layer.stride[0]
            + 1
        )
    )
    w_max = int(
        np.floor(
            (w_input + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1)
            / layer.stride[1]
            + 1
        )
    )
    return h_max, w_max


def get_unfold_params(layer):
    return dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)


def avg_value_by_layer(full):
    """
    return average value per layer across training

    **full** is a list of lists where the outer list is each snapshot through training or
    minibatch etc and each inner list is the value for each node in the network across layers
    of a particular measurement

    For example:
    num_epochs = 1000
    nodes_per_layer = [50, 40, 30, 20]
    len(full) == 1000
    len(full[i]) == 4 ... for all i
    [f.shape for f in full[i]] = [50, 40, 30, 20] ... for all i

    this method will return a tensor of size (num_layers, num_epochs) of the average value (for
    whatever value is in **full**) for each list/list
    """
    num_epochs = len(full)
    num_layers = len(full[0])
    avg_full = torch.zeros((num_layers, num_epochs))
    for layer in range(num_layers):
        avg_full[layer, :] = torch.tensor([torch.mean(f[layer]) for f in full])
    return avg_full.cpu()


def value_by_layer(full: List[List[torch.Tensor]], layer: int) -> torch.Tensor:
    """
    return all value measurements for a particular layer from **full**

    **full** is a list of lists where the outer list is each snapshot through training or
    minibatch etc and each inner list is the value for each node in the network across layers

    this method will return just the part of **full** corresponding to the layer indexed
    by **layer** as a tensor of shape (num_epochs, num_nodes)

    see ``avg_value_by_layer`` for a little more explanation
    """
    return torch.cat([f[layer].view(1, -1) for f in full], dim=0).cpu()


def condense_values(full: List[List[List[torch.Tensor]]]) -> List[torch.Tensor]:
    """
    condense List[List[List[Tensor]]] representing some value measured across networks, batches, and layers, for each node in the layer

    returns list of #=num_layers tensors, where each tensor has shape (num_networks, num_batches, num_nodes_per_layer)

    full should be a list of list of lists
    the first list should have length = number of networks
    the second list should have length = number of batches
    the third list should have length = number of layers in the network (this has to be the same for each network!)
    the tensor should have shape = number of nodes in this layer (also must be the same for each network) (or can be anything as long as consistent across layers)
    """
    num_layers = len(full[0][0])
    return [
        torch.stack([value_by_layer(value, layer) for value in full])
        for layer in range(num_layers)
    ]


def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))


def named_transpose(list_of_lists):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(list, zip(*list_of_lists))


def ptp(tensor, dim=None, keepdim=False):
    """
    simple method for measuring range of tensor on requested dimension or on all data
    """
    if dim is None:
        return tensor.max() - tensor.min()
    return tensor.max(dim, keepdim).values - tensor.min(dim, keepdim).values


def rms(tensor, dim=None, keepdim=False):
    """simple method for measuring root-mean-square on requested dimension or on all data in tensor"""
    if dim is None:
        return torch.sqrt(torch.mean(tensor**2))
    return torch.sqrt(torch.mean(tensor**2, dim=dim, keepdim=keepdim))


def compute_stats_by_type(tensor, num_types, dim, method="var"):
    """
    helper method for returning the mean and variance across a certain dimension
    where multiple types are concatenated on that dimension

    for example, suppose we trained 2 networks each with 3 sets of parameters
    and concatenated the loss in a tensor like [set1-loss-net1, set1-loss-net2, set2-loss-net1, ...]
    then this would contract across the nets from each set and return the mean and variance
    """
    num_on_dim = tensor.size(dim)
    num_per_type = int(num_on_dim / num_types)
    tensor_by_type = tensor.unsqueeze(dim)
    expand_shape = list(tensor_by_type.shape)
    expand_shape[dim + 1] = num_per_type
    expand_shape[dim] = num_types
    tensor_by_type = tensor_by_type.view(expand_shape)
    type_means = torch.mean(tensor_by_type, dim=dim + 1)
    if method == "var":
        type_dev = torch.var(tensor_by_type, dim=dim + 1)
    elif method == "std":
        type_dev = torch.std(tensor_by_type, dim=dim + 1)
    elif method == "se":
        type_dev = torch.std(tensor_by_type, dim=dim + 1) / np.sqrt(num_per_type)
    elif method == "range":
        type_dev = ptp(tensor_by_type, dim=dim + 1)
    else:
        raise ValueError(f"Method ({method}) not recognized.")

    return type_means, type_dev


def weighted_average(data, weights, dim, keepdim=False, ignore_nan=False):
    """
    take the weighted average of **data** on a certain dimension with **weights**

    weights should be a nonnegative vector that broadcasts into data
    avg = sum_i(data_i * weight_i, dim) / sum_i(weight_i, dim)

    if ignore_nan=True, (default=False), will ignore nans in weighted average
    """
    assert data.ndim == weights.ndim, "data and weights must have same number of dimensions"
    assert torch.all(weights[~torch.isnan(weights)] >= 0), "weights must be nonnegative"

    for d in dim if check_iterable(dim) else [dim]:
        assert data.size(d) == weights.size(
            d
        ), f"data and weights must have same size in averaging dimensions (data.size({d})={data.size(d)}, (weight.size({d})={weights.size(d)}))"

    # use normal sum if not ignore nan, otherwise use nansum
    sum = torch.nansum if ignore_nan else torch.sum

    # make sure nans are in the same place in weights and data for accurate division by total weight
    if ignore_nan:
        weights = weights.expand(data.size())
        weights = torch.masked_fill(weights, torch.isnan(data), torch.nan)

    # numerator & denominator of weighted average
    numerator = sum(data * weights, dim=dim, keepdim=keepdim)
    denominator = sum(weights, dim=dim, keepdim=keepdim)

    # return weighted average
    return numerator / denominator


def fgsm_attack(image, epsilon, data_grad, transform, sign):
    """update an image with fast-gradient sign method"""
    warn("fgsm_attack is only going to be in utils temporarily!", DeprecationWarning, stacklevel=2)
    # Collect the element-wise sign of the data gradient
    if sign:
        data_grad = data_grad.sign()
    else:
        data_grad = data_grad.clone()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = transform(perturbed_image)
    # Return the perturbed image
    return perturbed_image


def str2bool(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ("true", "1"):
        return True
    elif str.lower() in ("false", "0"):
        return False
    else:
        raise TypeError("Boolean type expected")


def save_checkpoint(nets, optimizers, results, path):
    """
    Method for saving checkpoints for networks throughout training.
    """
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {
        f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)
    }
    checkpoint = results | multi_model_ckpt | multi_optimizer_ckpt
    torch.save(checkpoint, path)


def load_checkpoints(nets, optimizers, device, path):
    """
    Method for loading presaved checkpoint during training.
    TODO: device handling for passing between gpu/cpu
    """

    if device == "cpu":
        checkpoint = torch.load(path, map_location=device)
    elif device == "cuda":
        checkpoint = torch.load(path)

    net_ids = sorted([key for key in checkpoint if key.startswith("model_state_dict")])
    opt_ids = sorted([key for key in checkpoint if key.startswith("optimizer_state_dict")])
    assert all(
        [oi.split("_")[-1] == ni.split("_")[-1] for oi, ni in zip(opt_ids, net_ids)]
    ), "nets and optimizers cannot be matched up from checkpoint"

    [net.load_state_dict(checkpoint.pop(net_id)) for net, net_id in zip(nets, net_ids)]
    [opt.load_state_dict(checkpoint.pop(opt_id)) for opt, opt_id in zip(optimizers, opt_ids)]

    if device == "cuda":
        [net.to(device) for net in nets]

    return nets, optimizers, checkpoint


def match_git(path):
    """simple method for determining if a path is a git-related file or directory"""
    if ".git" in path:
        return True
    return False


def compress_directory(output_path, directory_path=None):
    """send an entire directory to a zip file at output_path, using .gitignore and ignoring .git files"""
    if directory_path is None:
        # relative to utils -- this is the main repo path
        directory_path = os.path.dirname(os.path.abspath(__file__)) + "/.."

    # Parse .gitignore file
    gitignore_path = os.path.join(directory_path, ".gitignore")
    matches = parse_gitignore(gitignore_path)

    # Prepare list for copying files
    files_to_copy = []
    archive_names = []
    for dirpath, dirnames, files in os.walk(directory_path):
        if matches(dirpath) or match_git(dirpath):
            # clear any files from within this path
            dirnames[:] = []
        else:
            # Filter files based on .gitignore rules (and don't save any .git files)
            keep_files = [f for f in files if not matches(f) and not match_git(f)]
            # Make full path
            full_files = [os.path.join(dirpath, f) for f in keep_files]
            for file in full_files:
                # Add file to the copy list
                files_to_copy.append(file)
                archive_names.append(os.path.relpath(file, directory_path))

    # create zip file
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # go through directory
        for file, name in zip(files_to_copy, archive_names):
            zipf.write(file, arcname=name)
