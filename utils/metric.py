import time
import torch


def accuracy(preds, labels):
    """
    The input and output tensor shapes:
       input: [N, chn, dim1, dim2]
       output: [N, chn, dim1, dim2]
    """
    
    batch_len = preds.shape[0]     # batch size
    
    """
    Simply calculate the accuracy of the prediction
    https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
    """
    num_correct = (preds * labels).sum()*1.0 / batch_len
    return num_correct
    
    
class AverageMeter(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = float(self.sum) / self.count
        
    
class Timer(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.interval = 0
        self.time = time.time()
        
    def value(self):
        return time.time() - self.time
        
    def tic(self):
        self.time = time.time()
        
    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval