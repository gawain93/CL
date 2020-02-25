#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:15:38 2020

@author: zhi
"""


import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(nn.Linear(self.in_dim, hidden_dim),
                                    #nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    #nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True))
        self.last = nn.Linear(hidden_dim, out_dim)
        
    def feature(self, x):
        x = self.linear(x.view(-1, self.in_dim))
        return x
    
    def logits(self, x):
        x =self.last(x)
        return x
    
    def forward(self, x):
        feature(x)
        logits(x)
        return x    
    

def MLP100():
    return MLP(hidden_dim=100)


def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    return MLP(hidden_dim=1000)