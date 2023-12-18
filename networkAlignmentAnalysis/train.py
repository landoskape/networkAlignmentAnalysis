import time
from tqdm import tqdm
import torch
from networkAlignmentAnalysis.utils import transpose_list


def train(nets, optimizers, dataset, **parameters):
    """method for training network on supervised learning problem"""

    # do some variable checks
    if not(isinstance(nets, list)): nets = [nets]
    if not(isinstance(optimizers, list)): optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and optimizers need to be equal length lists"

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    use_train = parameters.get('train_set', True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataset.train_loader)*parameters['num_epochs']
    track_loss = torch.zeros((num_steps, num_nets))
    track_performance = []
    
    # --- optional analyses ---
    measure_alignment = parameters.get('alignment', True)
    measure_delta_weights = parameters.get('delta_weights', False)

    # measure alignment throughout training
    if measure_alignment: 
        alignment = []

    # measure weight norm throughout training
    if measure_delta_weights: 
        delta_weights = []
        init_weights = [net.get_alignment_weights() for net in nets]

    # --- training loop ---
    for epoch in range(parameters['num_epochs']):
        print('Epoch: ', epoch)

        for idx, batch in tqdm(enumerate(dataloader)):
            cidx = epoch*len(dataloader) + idx
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

            track_loss[cidx] = torch.tensor([l.item() for l in loss])
            track_performance.append([dataset.measure_performance(output, labels) for output in outputs])

            if measure_alignment:
                # Measure alignment if requested
                alignment.append([net.measure_alignment(images, precomputed=True, method='alignment') 
                                  for net in nets])
            
            if measure_delta_weights:
                # Measure change in weights if requested
                delta_weights.append([net.compare_weights(init_weight)
                                      for net, init_weight in zip(nets, init_weights)])
    
    results = {
        'loss': track_loss,
        'performance': transpose_list(track_performance),
    }
    
    # add optional analyses
    if measure_alignment: 
        results['alignment'] = transpose_list(alignment)
    if measure_delta_weights:
        results['delta_weights'] = transpose_list(delta_weights)

    return results


@torch.no_grad()
def test(net, dataset, **parameters):
    """method for testing network on supervised learning problem"""

    # retrieve requested dataloader from dataset
    use_test = parameters.get('test_set', True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # Performance Measurements
    total_loss = 0
    num_correct = 0
    num_attempted = 0
    alignment = []

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        minibatch_size = images.size(0)

        # Perform forward pass
        outputs = net(images, store_hidden=True)

        # Perform backward pass & optimization
        total_loss += dataset.measure_loss(outputs, labels).item()
        num_correct += dataset.measure_performance(outputs, labels, percentage=False)
        num_attempted += minibatch_size

        # Measure Integration
        alignment.append(net.measure_alignment(images, precomputed=True, method='alignment'))
    
    results = {
        'loss': total_loss / num_attempted,
        'accuracy': 100 * num_correct / num_attempted,
        'alignment': alignment,
    }

    return results

