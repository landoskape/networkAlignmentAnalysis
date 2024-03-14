from copy import copy, deepcopy
from tqdm import tqdm
import torch
from networkAlignmentAnalysis.utils import (
    transpose_list,
    condense_values,
    test_nets,
    train_nets,
    save_checkpoint,
    smart_pca,
    expected_alignment_distribution,
)


@train_nets
def train(nets, optimizers, dataset, **parameters):
    """method for training network on supervised learning problem"""

    # input argument checks
    if not (isinstance(nets, list)):
        nets = [nets]
    if not (isinstance(optimizers, list)):
        optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and optimizers need to be equal length lists"

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataset.train_loader) * parameters["num_epochs"]

    # --- optional W&B logging ---
    run = parameters.get("run")

    # --- optional analyses ---
    measure_alignment = parameters.get("alignment", True)
    measure_delta_weights = parameters.get("delta_weights", False)
    measure_frequency = parameters.get("frequency", 1)
    compare_expected = parameters.get("compare_expected", False)

    # --- optional training method: manual shaping with eigenvectors ---
    manual_shape = parameters.get("manual_shape", False)  # true or False, whether to do this
    # frequency of manual shape (similar to measure_frequency)
    # if positive, by minibatch, if negative, by epoch
    manual_frequency = parameters.get("manual_frequency", -1)
    manual_transforms = parameters.get("manual_transforms", None)  # len()==len(nets) callable methods
    manual_layers = parameters.get("manual_layers", None)  # index to which layers

    # --- create results dictionary if not provided and handle checkpoint info ---
    results = parameters.get("results", False)
    num_complete = parameters.get("num_complete", 0)
    save_ckpt, freq_ckpt, path_ckpt, dev = parameters.get("save_checkpoints", (False, 1, "", ""))
    if not results:
        # initialize dictionary for storing performance across epochs
        results = {
            "loss": torch.zeros((num_steps, num_nets)),
            "accuracy": torch.zeros((num_steps, num_nets)),
        }

        # measure alignment throughout training
        if measure_alignment:
            results["alignment"] = []

        # measure weight norm throughout training
        if measure_delta_weights:
            results["delta_weights"] = []
            results["init_weights"] = [net.get_alignment_weights() for net in nets]

        # compare true alignment distribution to expected distribution (according to Fiete alignment definition)
        if compare_expected:
            calign_bins = torch.linspace(0, 1, 301)
            results["compare_alignment_bins"] = calign_bins
            results["compare_alignment_expected"] = []
            results["compare_alignment_observed"] = []

    # If loaded from checkpoint but running more epochs than initialized for.
    elif results["loss"].shape[0] < num_steps:
        add_steps = num_steps - results["loss"].shape[0]
        assert (add_steps / (parameters["num_epochs"] - num_complete)) == len(
            dataset.train_loader
        ), "Number of new steps needs to multiple of epochs and num minibatches"
        results["loss"] = torch.vstack((results["loss"], torch.zeros((add_steps, num_nets))))
        results["accuracy"] = torch.vstack((results["accuracy"], torch.zeros((add_steps, num_nets))))

    if num_complete > 0:
        print("resuming training from checkpoint on epoch", num_complete)

    # --- training loop ---
    for epoch in tqdm(range(num_complete, parameters["num_epochs"]), desc="training epoch"):
        for idx, batch in enumerate(tqdm(dataloader, desc="minibatch", leave=False)):
            cidx = epoch * len(dataloader) + idx
            images, labels = dataset.unwrap_batch(batch)

            # Zero the gradients
            for opt in optimizers:
                opt.zero_grad()

            # Perform forward pass
            outputs = [net(images, store_hidden=True) for net in nets]

            # Perform backward pass & optimization
            loss = [dataset.measure_loss(output, labels) for output in outputs]
            for l, opt in zip(loss, optimizers):
                l.backward()
                opt.step()

            results["loss"][cidx] = torch.tensor([l.item() for l in loss])
            results["accuracy"][cidx] = torch.tensor([dataset.measure_accuracy(output, labels) for output in outputs])

            if idx % measure_frequency == 0:
                if measure_alignment:
                    # Measure alignment if requested
                    results["alignment"].append([net.measure_alignment(images, precomputed=True, method="alignment") for net in nets])

                if measure_delta_weights:
                    # Measure change in weights if requested
                    results["delta_weights"].append([net.compare_weights(init_weight) for net, init_weight in zip(nets, results["init_weights"])])

                if compare_expected:
                    # Measure distribution of alignment, compare with expected given "Alignment" from Fiete definition
                    # NOTE: This isn't optimized because it measures alignment twice (see above)
                    c_alignment = [net.measure_alignment(images, precomputed=True, method="alignment") for net in nets]
                    c_inputs = [net.get_layer_inputs(images, precomputed=True) for net in nets]
                    c_inputs = [net._preprocess_inputs(cin) for net, cin in zip(nets, c_inputs)]
                    c_evals = [[smart_pca(c.T)[0] for c in cin] for cin in c_inputs]
                    c_dist = [[expected_alignment_distribution(ev, valid_rotation=False, bins=calign_bins)[0] for ev in c_eval] for c_eval in c_evals]
                    t_dist = [[torch.histogram(align.cpu(), bins=calign_bins, density=True)[0] for align in c_align] for c_align in c_alignment]
                    results["compare_alignment_expected"].append(c_dist)
                    results["compare_alignment_observed"].append(t_dist)

            if run is not None:
                run.log(
                    {f"losses/loss-{ii}": l.item() for ii, l in enumerate(loss)}
                    | {f"accuracies/accuracy-{ii}": dataset.measure_accuracy(output, labels) for ii, output in enumerate(outputs)}
                    # {f'alignments/alignment-{ii}': alignment for ii, alignment in enumerate(results['alignment'][-1])}
                    | {"batch": cidx}
                )

        if manual_shape:
            # only do it at the end of #=manual_frequency epochs (but not last)
            if ((epoch + 1) % manual_frequency == 0) and (epoch < parameters["num_epochs"] - 1):
                for net, transform in tqdm(zip(nets, manual_transforms), desc="manual shaping", leave=False):
                    # just use this minibatch for computing eigenfeatures
                    inputs, _ = net._process_collect_activity(dataset, train_set=False, with_updates=False, use_training_mode=False)
                    _, eigenvalues, eigenvectors = net.measure_eigenfeatures(inputs, with_updates=False)
                    idx_to_layer_lookup = {layer: idx for idx, layer in enumerate(net.get_alignment_layer_indices())}
                    eigenvalues = [eigenvalues[idx_to_layer_lookup[ml]] for ml in manual_layers]
                    eigenvectors = [eigenvectors[idx_to_layer_lookup[ml]] for ml in manual_layers]
                    net.shape_eigenfeatures(manual_layers, eigenvalues, eigenvectors, transform)

        if save_ckpt & (epoch % freq_ckpt == 0):
            save_checkpoint(
                nets,
                optimizers,
                results | {"prms": parameters, "epoch": epoch, "device": dev},
                path_ckpt,
            )

    # condense optional analyses
    for k in ["alignment", "delta_weights", "avgcorr", "fullcorr", "compare_alignment_expected", "compare_alignment_observed"]:
        if k not in results.keys():
            continue
        results[k] = condense_values(transpose_list(results[k]))

    return results


@torch.no_grad()
@test_nets
def test(nets, dataset, **parameters):
    """method for testing network on supervised learning problem"""

    # --- optional W&B logging ---
    run = parameters.get("run")

    # input argument checks
    if not (isinstance(nets, list)):
        nets = [nets]

    # preallocate variables and define metaparameters
    num_nets = len(nets)

    # retrieve requested dataloader from dataset
    use_test = not parameters.get("train_set", False)  # if train_set=True, use_test=False
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # Performance Measurements
    total_loss = [0 for _ in range(num_nets)]
    num_correct = [0 for _ in range(num_nets)]
    num_batches = 0

    # --- optional analyses ---
    measure_alignment = parameters.get("alignment", True)

    # measure alignment throughout training
    if measure_alignment:
        alignment = []

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)

        # Perform forward pass
        outputs = [net(images, store_hidden=True) for net in nets]

        # Performance Measurements
        for idx, output in enumerate(outputs):
            total_loss[idx] += dataset.measure_loss(output, labels).item()
            num_correct[idx] += dataset.measure_accuracy(output, labels)

        # Keep track of number of batches
        num_batches += 1

        # Measure Alignment
        if measure_alignment:
            alignment.append([net.measure_alignment(images, precomputed=True, method="alignment") for net in nets])

    results = {
        "loss": [loss / num_batches for loss in total_loss],
        "accuracy": [correct / num_batches for correct in num_correct],
    }

    if measure_alignment:
        results["alignment"] = condense_values(transpose_list(alignment))

    if run is not None:
        run.summary["test_loss"] = torch.mean(torch.tensor(results["loss"]))
        run.summary["test_accuracy"] = torch.mean(torch.tensor(results["accuracy"]))

    return results


@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer

    idx_alignment should be a list of the indices of alignment (sorted from lowest to highest)
    where len(idx_alignment)=num_layers_per_network and each element is a tensor such that
    idx_alignment[0].shape=(num_nodes_per_layer, num_networks)

    returns a fraction of indices to drop of highest, lowest, and random alignment
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low = [idx[:, :drop] for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0) for nodes, drop in zip(num_nodes, num_drop)]
    return idx_high, idx_low, idx_rand


@torch.no_grad()
@test_nets
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout

    takes as input a list of networks (usually trained) and a dataset, along with some
    experiment parameters (although there are defaults coded into the method)

    note that this only works when each network in the list has the same architecture!
    to analyze a group of networks with different architectures, run this function multiple
    times and concatenate the results.

    alignment, when provided is a list of the alignment measurement for each layer of
    each network. It is expected that each alignment list has the structure:
    len(alignment)=num_alignment_layers
    alignment[i].shape = (num_networks, num_batches, num_dimensions_per_layer)
    : see how the outputs of ``measure_alignment`` is handled in the ``train`` and ``test``
    functions of this module to understand how to structure it in that format.

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more nodes based on the highest, lowest,
    or random alignment. Can either do it for each layer separately or all togehter using
    the parameters['by_layer'] kwarg.
    """

    # input argument check
    if not (isinstance(nets, list)):
        nets = [nets]

    # get index to each alignment layer
    idx_dropout_layers = nets[0].get_alignment_layer_indices()

    # get alignment of networks if not provided
    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    # check if alignment has the right length (ie number of layers) (otherwise can't make assumptions about where the classification layer is)
    assert len(alignment) == len(idx_dropout_layers), "the number of layers in **alignment** doesn't correspond to the number of alignment layers"

    # don't dropout classification layer if included as an alignment layer
    classification_layer = nets[0].num_layers(all=True) - 1  # index to last layer in network
    if classification_layer in idx_dropout_layers:
        idx_dropout_layers.pop(-1)
        alignment.pop(-1)

    # get average alignment (across batches) and index of average alignment by node
    alignment = [torch.mean(align, dim=1) for align in alignment]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

    # preallocate tracker tensors
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get dropout indices for this fraction of dropouts
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, fraction)

            # do drop out for each layer (or across all depending on parameters)
            for layer in range(num_layers):
                if by_layer:
                    drop_high, drop_low, drop_rand = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [idx_dropout_layers[layer]]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = copy(idx_dropout_layers)

                # get output with targeted dropout
                out_high = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_high], drop_layer)[0] for idx, net in enumerate(nets)]
                out_low = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_low], drop_layer)[0] for idx, net in enumerate(nets)]
                out_rand = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_rand], drop_layer)[0] for idx, net in enumerate(nets)]

                # get loss with targeted dropout
                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                # get accuracy with targeted dropout
                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

                # add to storage tensors
                progdrop_loss_high[:, dropidx, layer] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer] += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer] += torch.tensor(loss_rand)

                progdrop_acc_high[:, dropidx, layer] += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer] += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer] += torch.tensor(acc_rand)

    results = {
        "progdrop_loss_high": progdrop_loss_high / num_batches,
        "progdrop_loss_low": progdrop_loss_low / num_batches,
        "progdrop_loss_rand": progdrop_loss_rand / num_batches,
        "progdrop_acc_high": progdrop_acc_high / num_batches,
        "progdrop_acc_low": progdrop_acc_low / num_batches,
        "progdrop_acc_rand": progdrop_acc_rand / num_batches,
        "dropout_fraction": drop_fraction,
        "by_layer": by_layer,
        "idx_dropout_layers": idx_dropout_layers,
    }

    return results


@torch.no_grad()
@test_nets
def eigenvector_dropout(nets, dataset, eigenvalues, eigenvectors, **parameters):
    """
    method for testing network on supervised learning problem with eigenvector dropout

    takes as input a list of networks (usually trained) and a dataset, along with the
    eigenvectors of the input to each alignment layer along with some experiment
    parameters (although there are defaults coded into the method)

    note that this only works when each network in the list has the same architecture!
    to analyze a group of networks with different architectures, run this function multiple
    times and concatenate the results.

    eigenvectors must have the following structure:
    a list of lists of eigenvectors to each layer for each network such that:
    len(eigenvectors) = num_networks
    len(eigenvectors[i]) = num_alignment_layers for all i
    eigenvectors[i][j].shape = (num_dim_input_to_j, num_dim_input_to_j)

    eigenvalues should have same structure except have vectors instead of square matrices

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more eigenvectors based on the highest,
    lowest, or random eigenvalues. Can either do it for each layer separately or all
    together using the parameters['by_layer'] kwarg.
    """

    # input argument check
    if not (isinstance(nets, list)):
        nets = [nets]

    # get index to each alignment layer
    idx_dropout_layers = nets[0].get_alignment_layer_indices()

    # check if alignment has the right length (ie number of layers) (otherwise can't make assumptions about where the classification layer is)
    assert all(
        [len(ev) == len(idx_dropout_layers) for ev in eigenvectors]
    ), "the number of layers in **eigenvectors** doesn't correspond to the number of alignment layers"
    assert all(
        [len(ev) == len(idx_dropout_layers) for ev in eigenvalues]
    ), "the number of layers in **eigenvalues** doesn't correspond to the number of alignment layers"

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

    # create index of eigenvalue for compatibility with get_dropout_indices
    idx_eigenvalue = [torch.fliplr(torch.tensor(range(0, ev.size(1))).expand(num_nets, -1)) for ev in eigenvectors[0]]

    # preallocate tracker tensors
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get dropout indices for this fraction of dropouts
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)

            # do drop out for each layer (or across all depending on parameters)
            for layer in range(num_layers):
                if by_layer:
                    drop_high, drop_low, drop_rand = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [idx_dropout_layers[layer]]
                    drop_evals = [[evals[layer]] for evals in eigenvalues]
                    drop_evecs = [[evecs[layer]] for evecs in eigenvectors]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = copy(idx_dropout_layers)
                    drop_evals = deepcopy(eigenvalues)
                    drop_evecs = deepcopy(eigenvectors)

                # get output with targeted dropout
                out_high = [
                    net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_high], drop_layer)[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                out_low = [
                    net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_low], drop_layer)[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                out_rand = [
                    net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_rand], drop_layer)[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                # get loss with targeted dropout
                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                # get accuracy with targeted dropout
                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

                # add to storage tensors
                progdrop_loss_high[:, dropidx, layer] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer] += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer] += torch.tensor(loss_rand)

                progdrop_acc_high[:, dropidx, layer] += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer] += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer] += torch.tensor(acc_rand)

    results = {
        "progdrop_loss_high": progdrop_loss_high / num_batches,
        "progdrop_loss_low": progdrop_loss_low / num_batches,
        "progdrop_loss_rand": progdrop_loss_rand / num_batches,
        "progdrop_acc_high": progdrop_acc_high / num_batches,
        "progdrop_acc_low": progdrop_acc_low / num_batches,
        "progdrop_acc_rand": progdrop_acc_rand / num_batches,
        "dropout_fraction": drop_fraction,
        "by_layer": by_layer,
        "idx_dropout_layers": idx_dropout_layers,
    }

    return results
