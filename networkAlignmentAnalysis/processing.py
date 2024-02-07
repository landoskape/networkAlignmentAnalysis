import os
from tqdm import tqdm
from . import train
from .utils import load_checkpoints

def train_networks(exp, nets, optimizers, dataset):
    """train and test networks"""
    # do training loop
    parameters = dict(
        train_set=True,
        num_epochs=exp.args.epochs,
        alignment=not(exp.args.no_alignment),
        delta_weights=exp.args.delta_weights,
        frequency=exp.args.frequency,
    )

    if exp.args.use_prev & os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, results = load_checkpoints(nets,
                                                        optimizers,
                                                        exp.args.device,
                                                        exp.get_checkpoint_path())
        for net in nets:
            net.train()

        parameters['num_complete'] = results['epoch'] + 1
        parameters['results'] = results
        print('loaded networks from previous checkpoint')

    if exp.args.save_ckpts:
        parameters['save_checkpoints'] = (True, 1, exp.get_checkpoint_path(), exp.args.device)

    print('training networks...')
    train_results = train.train(nets, optimizers, dataset, **parameters)

    # do testing loop
    print('testing networks...')
    parameters['train_set'] = False
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results

def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    perform a progressive dropout (of nodes) experiment
    alignment is optional, but will be recomputed if you've already measured it. You can provide it
    by setting: alignment=test_results['alignment'] if ``train_networks`` has already been run.
    """
    # do targeted dropout experiment
    print('performing targeted dropout...')
    dropout_parameters = dict(num_drops=exp.args.num_drops, by_layer=exp.args.dropout_by_layer, train_set=train_set)
    dropout_results = train.progressive_dropout(nets, dataset, alignment=alignment, **dropout_parameters)
    return dropout_results, dropout_parameters

def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    # measure eigenfeatures
    print('measuring eigenfeatures...')
    beta, eigvals, eigvecs, class_betas = [], [], [], []
    dataloader = dataset.train_loader if train_set else dataset.test_loader
    for net in tqdm(nets):
        eigenfeatures = net.measure_eigenfeatures(dataloader, with_updates=False)
        beta_by_class = net.measure_class_eigenfeatures(dataloader, eigenfeatures[2], rms=False, with_updates=False)
        beta.append(eigenfeatures[0])
        eigvals.append(eigenfeatures[1])
        eigvecs.append(eigenfeatures[2])
        class_betas.append(beta_by_class)

    # make it a dictionary
    return dict(beta=beta, eigvals=eigvals, eigvecs=eigvecs, class_betas=class_betas, class_names=dataloader.dataset.classes) 

def eigenvector_dropout(exp, nets, dataset, eigen_results, train_set=False):
    """
    do targeted eigenvector dropout with precomputed eigenfeatures
    """
    # do targeted dropout experiment
    print('performing targeted eigenvector dropout...')
    evec_dropout_parameters = dict(num_drops=exp.args.num_drops, by_layer=exp.args.dropout_by_layer, train_set=train_set)
    evec_dropout_results = train.eigenvector_dropout(nets, dataset, eigen_results['eigvals'], eigen_results['eigvecs'], **evec_dropout_parameters)
    return evec_dropout_results, evec_dropout_parameters