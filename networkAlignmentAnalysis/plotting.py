import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl

import torch
from .utils import compute_stats_by_type, named_transpose, transpose_list, rms


def plot_train_results(exp, train_results, test_results, prms):
    """
    plotting method for training trajectories and testing data
    """

    num_train_epochs = train_results["loss"].size(0)
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]

    print("getting statistics on run data...")
    plot_alignment = "alignment" in train_results
    if plot_alignment:
        alignment = torch.stack([torch.mean(align, dim=2) for align in train_results["alignment"]])

    cmap = mpl.colormaps["tab10"]

    train_loss_mean, train_loss_se = compute_stats_by_type(train_results["loss"], num_types=num_types, dim=1, method="se")
    train_acc_mean, train_acc_se = compute_stats_by_type(train_results["accuracy"], num_types=num_types, dim=1, method="se")

    if plot_alignment:
        align_mean, align_se = compute_stats_by_type(alignment, num_types=num_types, dim=1, method="se")

    test_loss_mean, test_loss_se = compute_stats_by_type(torch.tensor(test_results["loss"]), num_types=num_types, dim=0, method="se")
    test_acc_mean, test_acc_se = compute_stats_by_type(torch.tensor(test_results["accuracy"]), num_types=num_types, dim=0, method="se")

    print("plotting run data...")
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0] + idx, xOffset[1] + idx]

    # Make Training and Testing Performance Figure
    alpha = 0.3
    figdim = 3
    figratio = 2
    width_ratios = [figdim, figdim / figratio, figdim, figdim / figratio]

    fig, ax = plt.subplots(1, 4, figsize=(sum(width_ratios), figdim), width_ratios=width_ratios, layout="constrained")

    # plot loss results fot training and testing
    for idx, label in enumerate(labels):
        cmn = train_loss_mean[:, idx]
        cse = train_loss_se[:, idx]
        tmn = test_loss_mean[idx]
        tse = test_loss_se[idx]

        ax[0].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
        ax[0].fill_between(range(num_train_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))
        ax[1].plot(get_x(idx), [tmn] * 2, color=cmap(idx), label=label, lw=4)
        ax[1].plot([idx, idx], [tmn - tse, tmn + tse], color=cmap(idx), lw=1.5)

    ax[0].set_xlabel("Training Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[0].set_ylim(0, None)
    ylims = ax[0].get_ylim()
    ax[1].set_xticks(range(num_types))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Testing")
    ax[1].set_xlim(-0.5, num_types - 0.5)
    ax[1].set_ylim(ylims)

    # plot loss results fot training and testing
    for idx, label in enumerate(labels):
        cmn = train_acc_mean[:, idx]
        cse = train_acc_se[:, idx]
        tmn = test_acc_mean[idx]
        tse = test_acc_se[idx]

        ax[2].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
        ax[2].fill_between(range(num_train_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))
        ax[3].plot(get_x(idx), [tmn] * 2, color=cmap(idx), label=label, lw=4)
        ax[3].plot([idx, idx], [tmn - tse, tmn + tse], color=cmap(idx), lw=1.5)

    ax[2].set_xlabel("Training Epoch")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title("Training Accuracy")
    ax[2].set_ylim(0, 100)
    ax[3].set_xticks(range(num_types))
    ax[3].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[3].set_ylabel("Accuracy (%)")
    ax[3].set_title("Testing")
    ax[3].set_xlim(-0.5, num_types - 0.5)
    ax[3].set_ylim(0, 100)

    exp.plot_ready("train_test_performance")

    # Make Alignment Figure
    if plot_alignment:
        num_align_epochs = align_mean.size(2)
        num_layers = align_mean.size(0)
        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers * figdim, figdim), layout="constrained", sharex=True)
        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                cmn = align_mean[layer, idx] * 100
                cse = align_se[layer, idx] * 100
                ax[layer].plot(range(num_align_epochs), cmn, color=cmap(idx), label=label)
                ax[layer].fill_between(range(num_align_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))

        for layer in range(num_layers):
            ax[layer].set_ylim(0, None)
            ax[layer].set_xlabel("Training Epoch")
            ax[layer].set_ylabel("Alignment (%)")
            ax[layer].set_title(f"Layer {layer}")

        ax[0].legend(loc="lower right")

        exp.plot_ready("train_alignment_by_layer")


def plot_dropout_results(exp, dropout_results, dropout_parameters, prms, dropout_type="nodes"):
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val} - dropout {dropout_type}" for val in prms["vals"]]
    cmap = mpl.colormaps["Set1"]
    alpha = 0.3
    msize = 10
    figdim = 3

    num_layers = dropout_results["progdrop_loss_high"].size(2)
    names = ["From high", "From low", "Random"]
    num_exp = len(names)
    dropout_fraction = dropout_results["dropout_fraction"]
    by_layer = dropout_results["by_layer"]
    extra_name = "by_layer" if by_layer else "all_layers"
    extra_name += dropout_type

    # Get statistics across each network type for progressive dropout experiment
    print("measuring statistics on dropout analysis...")
    loss_mean_high, loss_se_high = compute_stats_by_type(dropout_results["progdrop_loss_high"], num_types=num_types, dim=0, method="se")
    loss_mean_low, loss_se_low = compute_stats_by_type(dropout_results["progdrop_loss_low"], num_types=num_types, dim=0, method="se")
    loss_mean_rand, loss_se_rand = compute_stats_by_type(dropout_results["progdrop_loss_rand"], num_types=num_types, dim=0, method="se")

    acc_mean_high, acc_se_high = compute_stats_by_type(dropout_results["progdrop_acc_high"], num_types=num_types, dim=0, method="se")
    acc_mean_low, acc_se_low = compute_stats_by_type(dropout_results["progdrop_acc_low"], num_types=num_types, dim=0, method="se")
    acc_mean_rand, acc_se_rand = compute_stats_by_type(dropout_results["progdrop_acc_rand"], num_types=num_types, dim=0, method="se")

    # Contract into lists for looping through to plot
    loss_mean = [loss_mean_high, loss_mean_low, loss_mean_rand]
    loss_se = [loss_se_high, loss_se_low, loss_se_rand]
    acc_mean = [acc_mean_high, acc_mean_low, acc_mean_rand]
    acc_se = [acc_se_high, acc_se_low, acc_se_rand]

    print("plotting dropout results...")
    # Plot Loss for progressive dropout experiment
    fig, ax = plt.subplots(
        num_layers,
        num_types,
        figsize=(num_types * figdim, num_layers * figdim),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    ax = np.reshape(ax, (num_layers, num_types))

    for idx, label in enumerate(labels):
        for layer in range(num_layers):
            for iexp, name in enumerate(names):
                cmn = loss_mean[iexp][idx, :, layer]
                cse = loss_se[iexp][idx, :, layer]
                ax[layer, idx].plot(
                    dropout_fraction,
                    cmn,
                    color=cmap(iexp),
                    marker=".",
                    markersize=msize,
                    label=name,
                )
                ax[layer, idx].fill_between(dropout_fraction, cmn + cse, cmn - cse, color=(cmap(iexp), alpha))

            if layer == 0:
                ax[layer, idx].set_title(label)

            if layer == num_layers - 1:
                ax[layer, idx].set_xlabel("Dropout Fraction")
                ax[layer, idx].set_xlim(0, 1)

            if idx == 0:
                ax[layer, idx].set_ylabel("Loss w/ Dropout")

            if iexp == num_exp - 1:
                ax[layer, idx].legend(loc="best")

    exp.plot_ready("prog_dropout_" + extra_name + "_loss")

    fig, ax = plt.subplots(
        num_layers,
        num_types,
        figsize=(num_types * figdim, num_layers * figdim),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    ax = np.reshape(ax, (num_layers, num_types))

    for idx, label in enumerate(labels):
        for layer in range(num_layers):
            for iexp, name in enumerate(names):
                cmn = acc_mean[iexp][idx, :, layer]
                cse = acc_se[iexp][idx, :, layer]
                ax[layer, idx].plot(
                    dropout_fraction,
                    cmn,
                    color=cmap(iexp),
                    marker=".",
                    markersize=msize,
                    label=name,
                )
                ax[layer, idx].fill_between(dropout_fraction, cmn + cse, cmn - cse, color=(cmap(iexp), alpha))

            ax[layer, idx].set_ylim(0, 100)

            if layer == 0:
                ax[layer, idx].set_title(label)

            if layer == num_layers - 1:
                ax[layer, idx].set_xlabel("Dropout Fraction")
                ax[layer, idx].set_xlim(0, 1)

            if idx == 0:
                ax[layer, idx].set_ylabel("Accuracy w/ Dropout")

            if iexp == num_exp - 1:
                ax[layer, idx].legend(loc="best")

    exp.plot_ready("prog_dropout_" + extra_name + "_accuracy")


def plot_eigenfeatures(exp, results, prms):
    """method for plotting results related to eigen-analysis"""
    beta, eigvals, class_betas, class_names = (
        results["beta"],
        results["eigvals"],
        results["class_betas"],
        results["class_names"],
    )
    beta = [[torch.abs(b) for b in net_beta] for net_beta in beta]
    class_betas = [[rms(cb, dim=2) for cb in net_class_beta] for net_class_beta in class_betas]

    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]
    class_cmap = mpl.colormaps["viridis"].resampled(len(class_names))

    print("measuring statistics of eigenfeature analyses...")

    # shape wrangling
    beta = [torch.stack(b) for b in transpose_list(beta)]
    eigvals = [torch.stack(ev) for ev in transpose_list(eigvals)]
    class_betas = [torch.stack(cb) for cb in transpose_list(class_betas)]

    # normalize to relative values
    beta = [b / b.sum(dim=2, keepdim=True) for b in beta]
    eigvals = [ev / ev.sum(dim=1, keepdim=True) for ev in eigvals]
    class_betas = [cb / cb.sum(dim=2, keepdim=True) for cb in class_betas]

    # reuse these a few times
    statprms = lambda method: dict(num_types=num_types, dim=0, method=method)

    # get mean and variance eigenvalues for each layer for each network type
    mean_evals, var_evals = named_transpose([compute_stats_by_type(ev, **statprms("var")) for ev in eigvals])

    # get sorted betas (sorted within each neuron)
    sorted_beta = [torch.sort(b, descending=True, dim=2).values for b in beta]

    # get mean / se beta for each layer for each network type
    mean_beta, se_beta = named_transpose([compute_stats_by_type(b, **statprms("var")) for b in beta])
    mean_sorted, se_sorted = named_transpose([compute_stats_by_type(b, **statprms("var")) for b in sorted_beta])
    mean_class_beta, se_class_beta = named_transpose([compute_stats_by_type(cb, **statprms("var")) for cb in class_betas])

    print("plotting eigenfeature results...")
    figdim = 3
    alpha = 0.3
    num_layers = len(mean_beta)
    fig, ax = plt.subplots(2, num_layers, figsize=(num_layers * figdim, figdim * 2), layout="constrained")

    for layer in range(num_layers):
        num_input = mean_evals[layer].size(1)
        num_nodes = mean_beta[layer].size(1)
        for idx, label in enumerate(labels):
            mn_ev = mean_evals[layer][idx]
            se_ev = var_evals[layer][idx]
            mn_beta = torch.mean(mean_beta[layer][idx], dim=0)
            se_beta = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
            mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
            se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)
            ax[0, layer].plot(
                range(num_input),
                mn_ev,
                color=cmap(idx),
                linestyle="--",
                label="eigvals" if idx == 0 else None,
            )
            ax[0, layer].plot(range(num_input), mn_beta, color=cmap(idx), label=label)
            ax[0, layer].fill_between(range(num_input), mn_beta + se_beta, mn_beta - se_beta, color=(cmap(idx), alpha))
            ax[1, layer].plot(range(num_input), mn_sort, color=cmap(idx), label=label)
            ax[1, layer].fill_between(range(num_input), mn_sort + se_sort, mn_sort - se_sort, color=(cmap(idx), alpha))

            ax[0, layer].set_xscale("log")
            ax[1, layer].set_xscale("log")
            ax[0, layer].set_xlabel("Input Dimension")
            ax[1, layer].set_xlabel("Sorted Input Dim")
            ax[0, layer].set_ylabel("Relative Eigval & Beta")
            ax[1, layer].set_ylabel("Relative Beta (Sorted)")
            ax[0, layer].set_title(f"Layer {layer}")
            ax[1, layer].set_title(f"Layer {layer}")

            if layer == num_layers - 1:
                ax[0, layer].legend(loc="best")
                ax[1, layer].legend(loc="best")

    exp.plot_ready("eigenfeatures")

    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers * figdim, figdim), layout="constrained")

    for layer in range(num_layers):
        num_input = mean_evals[layer].size(1)
        num_nodes = mean_beta[layer].size(1)
        for idx, label in enumerate(labels):
            mn_ev = mean_evals[layer][idx]
            se_ev = var_evals[layer][idx]
            mn_beta = torch.mean(mean_beta[layer][idx], dim=0)
            se_beta = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
            mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
            se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)
            ax[layer].plot(
                range(num_input),
                mn_ev,
                color=cmap(idx),
                linestyle="--",
                label="eigvals" if idx == 0 else None,
            )
            ax[layer].plot(range(num_input), mn_beta, color=cmap(idx), label=label)
            ax[layer].fill_between(range(num_input), mn_beta + se_beta, mn_beta - se_beta, color=(cmap(idx), alpha))

            ax[layer].set_xscale("log")
            ax[layer].set_yscale("log")
            ax[layer].set_xlabel("Input Dimension")
            ax[layer].set_ylabel("Relative Eigval & Beta")
            ax[layer].set_title(f"Layer {layer}")

            if layer == num_layers - 1:
                ax[layer].legend(loc="best")

    exp.plot_ready("eigenfeatures_loglog")

    fig, ax = plt.subplots(
        num_types,
        num_layers,
        figsize=(num_layers * figdim, figdim * num_types),
        layout="constrained",
    )
    ax = np.reshape(ax, (num_types, num_layers))
    for layer in range(num_layers):
        num_input = mean_evals[layer].size(1)
        for idx, label in enumerate(labels):
            for idx_class, class_name in enumerate(class_names):
                mn_data = mean_class_beta[layer][idx][idx_class]
                se_data = se_class_beta[layer][idx][idx_class]
                ax[idx, layer].plot(range(num_input), mn_data, color=class_cmap(idx_class), label=class_name)
                ax[idx, layer].fill_between(
                    range(num_input),
                    mn_data + se_data,
                    mn_data - se_data,
                    color=(class_cmap(idx_class), alpha),
                )

            ax[idx, layer].set_xscale("log")
            ax[idx, layer].set_yscale("linear")
            ax[idx, layer].set_xlabel("Input Dimension")
            if layer == 0:
                ax[idx, layer].set_ylabel(f"{label}\nClass Loading (RMS)")
            if idx == 0:
                ax[idx, layer].set_title(f"Layer {layer}")

            if layer == num_layers - 1:
                ax[idx, layer].legend(loc="upper right", fontsize=6)

    exp.plot_ready("class_eigenfeatures")


def plot_adversarial_results(exp, eigen_results, adversarial_results, prms):
    accuracy, beta, eigvals = (
        adversarial_results["accuracy"],
        adversarial_results["betas"],
        eigen_results["eigvals"],
    )
    epsilons, use_sign = adversarial_results["epsilons"], adversarial_results["use_sign"]

    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]

    print("measuring statistics of adversarial analyses...")

    # shape wrangling
    accuracy = torch.stack([torch.stack(acc) for acc in transpose_list(accuracy)])  # (num_epsilon, num_nets)
    eigvals = [torch.stack(ev) for ev in transpose_list(eigvals)]  # [(num_nets, dim_layer) for each layer]
    beta = [torch.stack(b) for b in beta]  # [(epsilon, num_nets, dim_layer) for each layer]

    # normalize to relative values
    beta = [b / b.sum(dim=2, keepdim=True) for b in beta]
    eigvals = [ev / ev.sum(dim=1, keepdim=True) for ev in eigvals]

    # reuse these a few times
    statprms = lambda dim, method: dict(num_types=num_types, dim=dim, method=method)

    # get mean and variance beta/eigenvalues for each layer for each network type
    mean_acc, se_acc = compute_stats_by_type(accuracy, **statprms(1, "var"))  # (num_epsilon, num_types)
    mean_beta, se_beta = named_transpose(
        [compute_stats_by_type(b, **statprms(1, "var")) for b in beta]
    )  # [(epsilon, num_types, dim_layer) for each layer]
    mean_evals, var_evals = named_transpose(
        [compute_stats_by_type(ev, **statprms(0, "var")) for ev in eigvals]
    )  # [(num_types, dim_layer) for each layer]

    print("plotting adversarial success results...")
    figdim = 3
    alpha = 0.3
    num_layers = len(mean_beta)
    fig, ax = plt.subplots(1, 1, figsize=(figdim, figdim), layout="constrained")
    for idx, label in enumerate(labels):
        ax.plot(epsilons, mean_acc[:, idx], color=cmap(idx), label=label)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy")
        ax.set_title("Adversarial Attack Success")
        ax.legend(loc="best")

    exp.plot_ready("adversarial_success")

    print("plotting adversarial structure...")
    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers * figdim, figdim), layout="constrained")
    for layer in range(num_layers):
        num_input = mean_beta[layer].size(2)
        for idx, label in enumerate(labels):
            ax[layer].plot(
                range(num_input),
                torch.nanmean(mean_beta[layer][:, idx], dim=0).detach(),
                color=cmap(idx),
                label=label,
            )
        ax[layer].set_xscale("log")
        ax[layer].set_xlabel("Input Dimension")
        ax[layer].set_ylabel("Average Component of Pertubation")
        ax[layer].set_title(f"Layer {layer}")
        if layer == num_layers - 1:
            ax[layer].legend(loc="best")

    exp.plot_ready("adversarial_structure")


def plot_rf(rf, width, alignment=None, alignBounds=None, showRFs=None, figSize=5):
    if showRFs is not None:
        rf = rf.reshape(rf.shape[0], -1)
        idxRandom = np.random.choice(range(rf.shape[0]), showRFs, replace=False)
        rf = rf[idxRandom, :]
    else:
        showRFs = rf.shape[0]
    # normalize
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(showRFs, width, width)
    # If necessary, create colormap
    if alignment is not None:
        cmap = mpl.cm.get_cmap("rainbow", rf.shape[0])
        cmapPeak = lambda x: cmap(x)
        if alignBounds is not None:
            alignment = alignment - alignBounds[0]
            alignment = alignment / (alignBounds[1] - alignBounds[0])
        else:
            alignment = alignment - alignment.min()
            alignment = alignment / alignment.max()

    # plotting
    n = int(np.ceil(np.sqrt(rf.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(figSize, figSize)

    N = 1000
    for i in tqdm(range(rf.shape[0])):
        ax = axes[i // n][i % n]
        if alignment is not None:
            vals = np.ones((N, 4))
            cAlignment = alignment[i].numpy()
            cPeak = cmapPeak(alignment[i].numpy())
            vals[:, 0] = np.linspace(0, cPeak[0], N)
            vals[:, 1] = np.linspace(0, cPeak[1], N)
            vals[:, 2] = np.linspace(0, cPeak[2], N)
            usecmap = mpl.colors.ListedColormap(vals)
            ax.imshow(rf[i], cmap=usecmap, vmin=-1, vmax=1)
        else:
            ax.imshow(rf[i], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for j in range(rf.shape[0], n * n):
        ax = axes[j // n][j % n]
        ax.imshow(np.ones_like(rf[0]) * -1, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig
