import torch
from ..metrics.meter import AverageMeter

def _message(meters):
    """ Build message for each meter in meters. """
    msgs = []
    for key, meter in meters.items():
        tmp = meter.avg
        if tmp.size() == ():    # No dimensions
            msg = f'({key}) {tmp.item():.5f}'
        else:
            arr = [f'{a.item():.5f}' for a in tmp]
            msg = f'({key}) ' + ', '.join(arr)
        msgs.append(msg)
        
    return ';'.join(msgs)


def train(model, dataloader, loss_fn, optimizer, metrics,
          device=torch.device('cpu'), every_k=2):
    """ Train 'model' for one epoch (one full iteration of 'dataloader')

    Parameters
    ----------
    model : nn.Model
        Neural Network Model
    dataloader :  torch.DataLoader
        data loader
    loss_fn : nn.Loss
        differentiable loss function, f(pred, gt)
    optimizer : torch.optim
        optimizer
    metrics : dict()
        dictionary of {name: metric}, where name is a string and and metric a
        metric function (of the form f(pred, gt) -> 1 x 1)
    device : torch.device
        selected device
    every_k : int
        how often to print metrics

    Returns
    -------
    AverageMeter
        train loss
    dict()
        dictionary of AverageMeter objects {name: Meter}, where name is a
        string and Meter an AverageMeter object.
    """
    n_batches = len(dataloader)
    train_loss = AverageMeter()
    meters = {key: AverageMeter(key) for key in metrics.keys()}
    
    model.train()
    for k, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass and Loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss
        train_loss.update(loss.item(), x.size(0))
        
        # Metrics
        for key, metric in metrics.items():
            meters[key].update(metric(pred, y).detach().cpu(), x.size(0))
        
        # Print only every k steps and in the last iteration
        if (k % every_k) == 0 or k == n_batches-1:
            s = _message(meters)
            print(f'\rTrain: {k+1:5}/{n_batches}, Loss: {train_loss.avg:.5f}, Metrics: {s}', end='', flush=True)


    print()
    return train_loss, meters


def validate(model, dataloader, metrics, device=torch.device('cpu'), every_k=2):
    """ Validate 'model' using 'dataloader' an the provided 'metrics'.

    Compute the passed metrics on the provided dataset.

    Parameters
    ----------
    model : nn.Model
        Neural Network Model
    dataloader :  torch.DataLoader
        data loader
    metrics : dict()
        dictionary of {name: metric}, where name is a string and and metric a
        metric function (of the form f(pred, gt) -> 1 x 1)
    device : torch.device
        selected device
    every_k : int
        how often to print metrics

    Returns
    -------
    dict()
        dictionary of AverageMeter objects {name: Meter}, where name is a
        string and Meter an AverageMeter object.
    """
    n_batches = len(dataloader)
    meters = {key: AverageMeter(key) for key in metrics.keys()}

    model.eval()
    with torch.no_grad():
        for k, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            
            # Metrics
            for key, metric in metrics.items():
                meters[key].update(metric(pred, y).detach().cpu(), x.size(0))
            
            # Print only every k steps and in the last iteration
            if (k % every_k) == 0 or k == n_batches-1:
                s = _message(meters)
                print(f'\rValidate: {k+1:5}/{n_batches}, Metrics: {s}', end='', flush=True)

    print()
    return meters
