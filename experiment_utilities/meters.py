# from pytorch imagenet example
class AverageMeter(object):
    """Computes and stores the average, max and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        if val < self.min:
            self.min = val
        if val > self.max:
            self.max = val
        self.count += n
        self.avg = self.sum / self.count