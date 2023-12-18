import time
from tqdm import tqdm
import torch
from networkAlignmentAnalysis.utils import transpose_list


def train(nets, optimizers, dataset, **parameters):
    """method for training network on supervised learning problem"""

    # input argument checks
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
            track_performance[cidx] = torch.tensor([dataset.measure_performance(output, labels) for output in outputs])

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
        'performance': track_performance,
    }
    
    # add optional analyses
    if measure_alignment: 
        results['alignment'] = transpose_list(alignment)
    if measure_delta_weights:
        results['delta_weights'] = transpose_list(delta_weights)

    return results


@torch.no_grad()
def test(nets, dataset, **parameters):
    """method for testing network on supervised learning problem"""

    # input argument checks
    if not(isinstance(nets, list)): nets = [nets]

    # preallocate variables and define metaparameters
    num_nets = len(nets)

    # retrieve requested dataloader from dataset
    use_test = not parameters.get('train_set', True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # Performance Measurements
    total_loss = [0 for _ in range(num_nets)]
    num_correct = [0 for _ in range(num_nets)]
    num_attempted = 0
    alignment = []

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        minibatch_size = images.size(0)

        # Perform forward pass
        outputs = [net(images, store_hidden=True) for net in nets]

        # Performance Measurements
        for idx, output in enumerate(outputs):
            total_loss[idx] += dataset.measure_loss(output, labels).item()
            num_correct[idx] += dataset.measure_performance(output, labels, percentage=False)
        
        # Keep track of number of inputs
        num_attempted += minibatch_size

        # Measure Integration
        alignment.append([net.measure_alignment(images, precomputed=True, method='alignment')
                          for net in nets])
    
    results = {
        'loss': [loss / num_attempted for loss in total_loss],
        'accuracy': [100 * correct / num_attempted for correct in num_correct],
        'alignment': transpose_list(alignment),
    }

    return results


def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout

    ... make me explain if this is still here!!! ...
    """

    # input argument checks
    if not(isinstance(nets, list)): nets = [nets]
    if alignment is not None:
        if not(isinstance(alignment, list)): 
            alignment = [alignment]
        assert len(alignment)==len(nets), "length of provided alignment data must be same as length of nets"
    else:
        alignment = test(nets, dataset, **parameters)['alignment']
        raise ValueError("need to average alignment!!!!")
    
    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters['num_drops']
    drop_fraction = torch.linspace(0,1,num_drops+2)[1:-1]
    by_layer = parameters['by_layer']
    num_layers = nets[0].num_layers() if by_layer else 1

    progdrop_loss_low = torch.zeros((num_nets, num_layers, num_drops))
    progdrop_loss_high = torch.zeros((num_nets, num_layers, num_drops))
    progdrop_loss_rand = torch.zeros((num_nets, num_layers, num_drops))
    progdrop_acc_low = torch.zeros((num_nets, num_layers, num_drops))
    progdrop_acc_high = torch.zeros((num_nets, num_layers, num_drops))
    progdrop_acc_rand = torch.zeros((num_nets, num_layers, num_drops))

    num_attempted = 0

    # retrieve requested dataloader from dataset
    use_test = not parameters.get('train_set', True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        minibatch_size = images.size(0)
        num_attempted += minibatch_size

        # do drop out for each layer (or across all depending on parameters)
        for layer in range(num_layers):
            for dropidx, fraction in enumerate(drop_fraction):
                


        # Perform forward pass
        outputs = [net(images, store_hidden=True) for net in nets]

        # Performance Measurements
        for idx, output in enumerate(outputs):
            total_loss[idx] += dataset.measure_loss(output, labels).item()
            num_correct[idx] += dataset.measure_performance(output, labels, percentage=False)
        
        # Keep track of number of inputs
        num_attempted += minibatch_size

        # Measure Integration
        alignment.append([net.measure_alignment(images, precomputed=True, method='alignment')
                          for net in nets])
    
    results = {}

    return results

progressBar = tqdm(testloader)
for batch in progressBar:
    images,label = batch
    images = images.to(DEVICE)
    label = label.to(DEVICE)
    
    for runidx in range(numRuns):
        for layer in range(numLayers):
            progressBar.set_description(f"RunIdx:{runidx+1}/{numRuns}, Layer:{layer+1}/{numLayers}")
                
            idxFinalAlignment = torch.argsort(alignLayer[layer][runidx,:,-1])
            for dropFrac in range(numDrops):
                num2look = int(dropFraction[dropFrac] * alignLayer[layer].shape[1])
                idxHi = idxFinalAlignment[-num2look:]
                idxLo = idxFinalAlignment[:num2look]
                
                # Get loss for progressive dropout of hi
                outputs = targetedDropout(nets[runidx], images, idxHi, layer)
                progDropLossHi[runidx,layer,dropFrac] += loss_function(outputs,label).item()
                output1 = torch.argmax(outputs,axis=1)
                progDropAccuracyHi[runidx,layer,dropFrac] += sum(output1==label).cpu()
                
                # Get loss for progressive dropout of low
                outputs = targetedDropout(nets[runidx], images, idxLo, layer)
                progDropLossLo[runidx,layer,dropFrac] += loss_function(outputs,label).item()
                output1 = torch.argmax(outputs,axis=1)
                progDropAccuracyLo[runidx,layer,dropFrac] += sum(output1==label).cpu()
                
# Normalize correctly
progDropLossLo = progDropLossLo/len(testloader)
progDropLossHi = progDropLossHi/len(testloader)
progDropAccuracyLo = 100*progDropAccuracyLo/len(testloader.dataset)
progDropAccuracyHi = 100*progDropAccuracyHi/len(testloader.dataset)