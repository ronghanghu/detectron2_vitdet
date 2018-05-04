from collections import deque
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MedianMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        window_size = 20
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.deque.append(val)
        self.series.append(val)
        self.count += n
        self.total += val * n

    @property
    def val(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return self.total / self.count if self.count else 0


