#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:38:44 2020

@author: zhi
"""


import torch.nn as nn
import torch.nn.functional as F
    
class LeNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz // 4
        self.n_feat = 50*feat_map_sz*feat_map_sz
        
        self.conv = nn.Sequential(nn.Conv2d(in_channel, 20, 5, padding=2),
                                  nn.BatchNorm2d(20),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2,2),
                                  nn.Conv2d(20, 50, 5, padding=2),
                                  nn.BatchNorm2d(50),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2,2))
        
        self.linear = nn.Sequential(nn.Linear(self.n_feat, 500),
                                    nn.BatchNorm2d(500),
                                    nn.ReLU(inplace=True))
        
        self.last = nn.Linear(500, out_dim)
        
    
    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x
    
    
    def logits(self, x):
        x = self.last(x)
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    

def LeNetC(out_dim=10):
    return LeNet(out_dim=out_dim, in_channel=3, img_sz=32)