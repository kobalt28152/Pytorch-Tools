class AverageMeter(object):
    """ Meter object.

    Given a metric comming from a batch value (e.g. (x_1 + ... + x_N)/N)
    compute and store the total average after inputing many batch
    metrics. """

    def __init__(self, name=None):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ 
        Parameters
        ----------
        val : torch.Tensor
            Average metric of a batch; e.g., (a1 + a2 + ... + an)/n
        n : int
            The batch size

        Using the previous paremters, then modify:

        sum : torch.Tensor
            The total sum of all batch elements
                sum += val * n -> a1 + a2 + a3 + ...
        count : int
            The total number of elements proccessed sofar
                count += n
        avg : torch.Tensor
            The total average
                avg = sum / count
        """
        self.val = val         # batch average
        self.sum += val * n    # total sum
        self.count += n        # total number of elements
        self.avg = self.sum / self.count    # total average

    def __str__(self):
        if self.count == 0:
            return f'Uninitialized meter'

        if self.avg.size() == ():    # No dimensions (scalar)
            s = f'{self.sum:.5f}/{self.count} = {self.avg:.5f}'
        else:
            tmp1 = ', '.join([f'{a:.5f}' for a in self.sum])
            tmp2 = ', '.join([f'{a:.5f}' for a in self.avg])
            s = f'({tmp1})/{self.count} = ({tmp2})'

        if self.name:
            return f'({self.name} meter) total avg: {s} '
        else:
            return f'Total avg: {s}'

    def __repr__(self):
        return self.__str__()
