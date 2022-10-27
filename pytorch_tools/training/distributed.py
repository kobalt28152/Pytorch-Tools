import torch

# Imports for Distributed Training 
# import torch.multiprocessing as mp                                         # Multi-processing lib
# from torch.utils.data.distributed import DistributedSampler                # Distributed Sampler
from torch.nn.parallel import DistributedDataParallel as DDP               # DDP model wrapper
# from torch.distributed import init_process_group, destroy_process_group    # Communication and syncrhonization

import operator
import os

from ..metrics.meter import AverageMeter



class Trainer:
    """ Trianer Class for Distributed GPU Training

    Basic trainer class that handles most of the standard training procedures.
    """

    def __init__(self, model, loss_fn, optimizer, dl_training, rank, world_size,
                 scheduler=None, dl_validate=None, metrics=None, metrics_cmp=None,
                 checkpoint='checkpoint.pt', save_every=10, print_every=2):
        """
        Parameters
        ----------
        model : nn.Module
            nn model
        loss_fn : function
            loss function
        optimizer : torch.optim.Optimizer
            optimizer
        dl_training : torch.utils.data.DataLoader
            dataloader for training data
        rank : int
            gpu/process identifier.
        scheduler : torch.optim.lr_scheduler
            (optional) scheduler to adjust the learning rate
        dl_validate : torch.utils.data.DataLoader
            (optional) dataloader for validation data
        metrics : dict{name: function}
            dictionary of metrics used for training and validation; e.g.
            {'bce': nn.BCEWithLogitsLoss. 'acc': Accuracy(), ...}
        metrics_cmp : dict{name: function}
            dictionary containing a comparison function for a metric; valid
            functions are: 'operator.lt' and 'operator.gt'. For instance:
            {'bce': operator.lt, 'iou': operator.gt, ...}
            NOTE. For metrics provided with a comparison function, the
            BEST SCORE is tracked and the parameters for the best scores are
            saved.
        checkpoint: str
            path to last checkpoint
        save_every : int
            save a checkpoint of the movel every `save_every' epochs
        print_every : int
            print training loss/metrics every 'print_every' steps
        """
        self.rank = rank                  # rank and world size
        self.world_size = world_size

        self.model = model.to(rank)       # Training related parameters
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dl_training = dl_training    # Dataloaders for training and validation
        self.dl_validate = dl_validate

        self.metrics = metrics

        # Only on rank 0:
        # - history of training/validation loss and metrics
        # - metrics comparison and best metric
        if rank == 0:
            self.hist_loss = []     # History of training loss (always given)

            # History for each training/validation metrics
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

        self.last_epoch = -1    # Last completed epoch; -1 -> not yet trained
        self.save_every = save_every
        self.print_every = print_every

        self.checkpoint = checkpoint

        # Load checkpoint if available
        if os.path.exists(checkpoint):
            self._load_checkpoint(checkpoint)
        else:
            print(f'({rank}): No checkpoint found. Starting from scratch')

        # Wrap model in DDP:
        # DDP broadcast state_dict() from rank 0 process to all other processes.
        # From now on to access state_dict(), we must write: model.module.state_dict()
        self.model = DDP(self.model, device_ids=[rank])

    def _load_checkpoint(self, path):
        # NOTE:
        # Loading a checkpoint must be done BEFORE wrapping model in DDP
        print(f'({self.rank}): Loading checkpoint: {path}')

        # Make sure that tensors are loaded to the correct device
        loc = f'cuda:{self.rank}'
        checkpoint = torch.load(path, map_location=loc)

        self.last_epoch = checkpoint['last_epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            self.scheduler = None

        # History only for rank 0 process
        if self.rank == 0:
            # History is stored in CPU
            checkpoint = torch.load(path, map_location='cpu')
            self.hist_loss = checkpoint['hist_loss']
            self.hist_train = checkpoint['hist_train']
            self.hist_valid = checkpoint['hist_valid']
            self.metrics_best = checkpoint['metrics_best']


    def _save_checkpoint(self, path):
        # NOTE:
        # At this point model has be wrapped by DDP. Hence, we call
        # model.module.state_dict()
        print(f'({self.rank}): Saving checkpoint: {path}')

        torch.save({
            'last_epoch': self.last_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'hist_loss': self.hist_loss,
            'hist_train': self.hist_train,
            'hist_valid': self.hist_valid,
            'metrics_best': self.metrics_best
        }, path)

    def _train(self):
        n_batches = len(self.dl_training)
        train_loss = AverageMeter()
        meters = {key: AverageMeter(key) for key in self.metrics.keys()}

        self.model.module.train()

        # NOTE: we must call the set_epoch() method at the beginning of each
        # epoch before creating the DataLoader iterator to make shuffling work
        # properly across multiple epochs.
        self.dl_training.sampler.set_epoch(self.last_epoch+1)
        for k, (x, y) in enumerate(self.dl_training):
            x = x.to(self.rank)
            y = y.to(self.rank)

            pred = self.model(x)            # Forward pass
            loss = self.loss_fn(pred, y)    # loss

            self.optimizer.zero_grad()      # Backward pass
            loss.backward()
            self.optimizer.step()

            # Loss and Metrics
            train_loss.update(loss.detach().cpu(), x.size(0))    # Save loss

            # For each metric update the corresponding meter
            for key, metric in self.metrics.items():
                meters[key].update(metric(pred, y).detach().cpu(), x.size(0))

            # Process 0: Print every 'print_every' steps and in the last iteration
            if (self.rank == 0) and ((k % self.print_every) == 0 or k == n_batches-1):
                s = self._message(meters)
                print(f'\r({self.rank}) Train: {k+1:5}/{n_batches}, Loss: {train_loss.avg:.5f}, Metrics: {s}', end='', flush=True)

        # Reduce loss
        loss_avg = self._reduce_meter(train_loss)
        # Reduce metrics
        meters_avg = {key: self._reduce_meter(meters[key]) for key in meters.keys()}

        # Process 0: print reduced loss and metrics
        if self.rank == 0:
            s = self._message(meters_avg)
            print(f'\nTotal | loss: {loss_avg}, Metrics: {s}\n', end='')

        return loss_avg, meters_avg

    def _validation(self):
        n_batches = len(self.dl_validate)
        meters = {key: AverageMeter(key) for key in self.metrics.keys()}

        self.model.module.eval()

        # No need to call set_epoch(); in validation shuffling is not necessary
        # self.dl_validate.sampler.set_epoch(self.last_epoch+1)
        for k, (x, y) in enumerate(self.dl_validate):
            x = x.to(self.rank)
            y = y.to(self.rank)

            with torch.no_grad():
                pred = self.model(x)

            # For each metric update the corresponding meter
            for key, metric in self.metrics.items():
                meters[key].update(metric(pred, y).detach().cpu(), x.size(0))

            # Process 0: Print only every k steps and in the last iteration
            if (self.rank == 0) and ((k % self.print_every) == 0 or k == n_batches-1):
                s = self._message(meters)
                print(f'\r({self.rank}) Validate: {k+1:5}/{n_batches}, Metrics: {s}', end='', flush=True)

        # Reduce metrics
        meters_avg = {key: self._reduce_meter(meters[key]) for key in meters.keys()}

        # Process 0: print reduced metrics
        if self.rank == 0:
            s = self._message(meters_avg)
            print(f'\nTotal | Metrics: {s}\n', end='')

        return meters_avg

    def _reduce_meter(self, meter):
        # An AverageMeter has two members: sum and count, that must be reduced:
        #     m1 -> sum = a1+a2+...+an; count = n
        #     m2 -> sum = b1+b2+...+bm; count = m
        # reduce to:
        #    sum = a1+a2+...+an + b1+b2+...+bn; count = n + m

        # Stack both 'sum' and 'count' in a single Tensor
        tensor = torch.hstack([meter.sum, torch.tensor(meter.count)]).to(self.rank)
        # Reduce
        torch.distributed.reduce(tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        tensor = tensor.to('cpu')

        # Return average
        # NOTE: only rank 0 returns the total average.
        return tensor[:-1]/tensor[-1]    # (a1+a2+...+an + b1+b2+...+bm) / (n+m)


    def _message(self, meters):
        # Build message for each meter in meters or for an array of tensors
        msgs = []
        for key, item in meters.items():
            tmp = item.avg if isinstance(item, AverageMeter) else item
            if tmp.size() == ():    # No dimensions
                msg = f'({key}) {tmp.item():.5f}'
            else:
                arr = [f'{a.item():.5f}' for a in tmp]
                msg = f'({key}) ' + ', '.join(arr)
            msgs.append(msg)

        return ';'.join(msgs)

    def _fill_hist(self, avgs, hist):
        for key in avgs.keys():
            hist[key].append(avgs[key])

    def _step(self):
        # One training step consists of:
        # 1. Train; 2. Validate; 3. Save loss and metrics
        # Train for one epoch
        loss_avg, met_tr_avg = self._train()

        # Process 0: save the training loss/metric scores
        if self.rank == 0:
            self.hist_loss.append(loss_avg)
            self._fill_hist(met_tr_avg, self.hist_train)

        # Validate if validation set provided
        if self.dl_validate:
            met_val_avg = self._validation()

            # Process 0: save the validation scores
            if self.rank == 0:
                self._fill_hist(met_val_avg, self.hist_valid)

        # If scheduler provided advance its step
        if self.scheduler: self.scheduler.step()

    def _validate_scores(self):
        # For each metric with compare function
        for key in self.metrics_cmp:
            # Metric could be a (n,)-tensor; take average in such case.
            cur = self.hist_valid[key][-1]
            cur = float(cur.mean()) if cur.dim() != 0 else float(cur)
            prev = self.metrics_best[key]

            # current value better than previou one, save model
            if self.metrics_cmp[key](cur, prev):
                self.metrics_best[key] = cur
                # path = os.path.join(self.path, f'model_best-{key}.pt')
                path =  f'model_best-{key}.pt'
                print(f'Saving parameters ({key}) at "{path}" ...')
                torch.save(self.model.state_dict(), path)

    def train(self, steps=1):
        init = self.last_epoch+1    # continue after the last trained epoch
        end = init + steps          # train for 'steps' epochs
        for epoch in range(init, end):
            # b_sz = len(next(iter(self.dl_training))[0])
            print(f"[GPU{self.rank}] Epoch {epoch} |  Steps: {len(self.dl_training)}\n", end='')

            self._step()
            self.last_epoch += 1    # Update lase_epoch after each step

            # Process 0: save a checkpoint every 'save_evry' iterations
            if self.rank == 0 and ((self.last_epoch+1) % self.save_every) == 0:
                self._save_checkpoint(self.checkpoint)

            # Proces 0: If validation set provided and metrics comparison
            # provided check validation scores and save the model with the best
            # score.
            if self.rank==0 and (self.dl_validate and self.metrics_cmp):
                self._validate_scores()
        
        # Process 0: save a checkpoint after finishing training
        if self.rank == 0:
            self._save_checkpoint(self.checkpoint)
