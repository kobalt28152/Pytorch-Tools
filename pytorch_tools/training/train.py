import torch

import operator
import os

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
        train_loss.update(loss.detach().cpu(), x.size(0))
        
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


class Trainer:
    """ Trianer Class 

    Basic trainer class that handles most of the standard training procedures.
    """

    def __init__(self, model, loss_fn, metrics, optimizer, dl_training, device,
                 scheduler=None, dl_validate=None, metrics_cmp=None, path=''):
        """
        Parameters
        ----------
        model : nn.Module
            nn model
        loss_fn : function
            loss function
        metrics : dict{name: function}
            dictionary of metrics used for training and validation; e.g.
            {'bce': nn.BCEWithLogitsLoss. 'acc': Accuracy(), ...}
        optimizer : torch.optim.Optimizer
            optimizer
        dl_training : torch.utils.data.DataLoader
            dataloader for training data
        device : torch.device
            device used for training
        scheduler : torch.optim.lr_scheduler
            (optional) scheduler to adjust the learning rate
        dl_validate : torch.utils.data.DataLoader
            (optional) dataloader for validation data
        metrics_cmp : dict{name: function}
            dictionary containing a comparison function for a metric; valid
            functions are: 'operator.lt' and 'operator.gt'. For instance:
            {'bce': operator.lt, 'iou': operator.gt, ...}
            NOTE. For metrics provided with a comparison function, the
            BEST SCORE is tracked and the parameters for the best scores are
            saved.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.dl_training = dl_training    # Dataloaders for training and validation
        self.dl_validate = dl_validate

        self.hist_loss = []     # History of training loss

        # History of training/validation metrics
        self.hist_train = {key: [] for key in metrics.keys()} if metrics else None
        self.hist_valid = {key: [] for key in metrics.keys()} if metrics else None

        # metrics compare: if comparison function provided for a meter, keep track
        # of the best score
        self.metrics_cmp = metrics_cmp
        if metrics_cmp:
            self.metrics_best = {key: float('inf') if item is operator.lt else -float('inf')
                                 for key, item in metrics_cmp.items()}
        else:
            self.metrics_best = None

        self.path = path
        self.last_epoch = -1    # Last completed epoch; -1 -> not trained yet

    def _fill_hist(self, meter, hist):
        for key in meter.keys():
            hist[key].append(meter[key].avg)

    def step(self):
        # Train for one epoch
        tr_loss, tr_met = train(self.model, self.dl_training, self.loss_fn, self.optimizer,
                                self.metrics, device=self.device)
        # Save loss and metric
        self.hist_loss.append(tr_loss.avg)
        self._fill_hist(tr_met, self.hist_train)

        # Validate if validation set provided
        if self.dl_validate:
            val_met = validate(self.model, self.dl_validate, self.metrics, device=self.device)
            self._fill_hist(val_met, self.hist_valid)

        # If scheduler provided advance its step
        if self.scheduler: self.scheduler.step()

    def train(self, steps=1):
        init = self.last_epoch+1    # continue after the last trained epoch
        end = init + steps          # train for 'steps' epochs
        for epoch in range(init, end):
            print(f'\nepoch: {epoch}')
            self.step()
            self.last_epoch += 1    # Update lase_epoch after each step

            # If validation set provided and metrics comparison provided
            # check validation scores and save the model with the best score
            if self.dl_validate and self.metrics_cmp:
                self.validate_scores()

    def validate_scores(self):
        # For each metric with compare function
        for key in self.metrics_cmp:
            # Metric could be a (n,)-tensor; take average in such case.
            cur = self.hist_valid[key][-1]
            cur = float(cur.mean()) if cur.dim() != 0 else float(cur)
            prev = self.metrics_best[key]

            # current value better than previou one, save model
            if self.metrics_cmp[key](cur, prev):
                self.metrics_best[key] = cur
                path = os.path.join(self.path, f'model_best-{key}.pt')
                print(f'Saving parameters ({key}) at "{path}" ...')
                torch.save(self.model.state_dict(), path)

    def save_checkpoint(self):
        save_path = os.path.join(self.path, f'checkpoint_{self.last_epoch}.tar')
        print(f'Saving checkpoint: {save_path}')

        torch.save({
            'last_epoch': self.last_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'hist_loss': self.hist_loss,
            'hist_train': self.hist_train,
            'hist_valid': self.hist_valid,
            'metrics_best': self.metrics_best
        }, save_path)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')

        checkpoint = torch.load(path)

        self.last_epoch = checkpoint['last_epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            self.scheduler = None

        self.hist_loss = checkpoint['hist_loss']
        self.hist_train = checkpoint['hist_train']
        self.hist_valid = checkpoint['hist_valid']
        self.metrics_best = checkpoint['metrics_best']
