# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:01:12 2020

@author: user
"""

from __future__ import print_functions
import torch
import torch.nn as nn

import models
from utils.metric import accuracy, Averagemeter, Timer

"""
A simpler version without multihead networks
Simple network architecture for classification tasks

 Task Names are globally unknown!!!!!
 
 TODO Test by training normal MNIST 
"""


class NormNN(nn.Module):
    """
    nn.Module is the base class for all Neural Networks in Pytorch
    Customed models should also subclass this class
    Modules can also contain other modules, allowing to nest them in a 
    tree structure.    
    """
    
    """
    Prototye of a learning pipeline of neural networks for multiple tasks.
    Steps:
          * Class initialization
          * Optimizer Initialization
          * Loss Defination
          * Model Updata
    """
    
    def __init__(self, Configs):
        """
        Configs: Dict of Configurations
        * Configurations for Networks:
           * Network type(string), LeNet, MLP...
           * Network name(string), LeNet100...
        * Configurations for Optimizers:
           * Optimizer type(string), SGD, Adam...
           * Momentum(float)
           * lr(float)
           * weight_decay(float)
           * lr_schedule([int])
           * reset_optim(bool)
        * Configurations for training:
           * cuda, if use cuda
        """
        super(NormNN, self).__init__()
        self.Configs = Configs
        self.model_type = Configs["model_type"]
        self.model_name = Configs["model_name"]
        self.model = self.createModel()
        
        self.optim_type = Configs["optim_type"]
        self.lr = Configs["lr"]
        self.lr_schedule = Configs['lr_schedule']
        
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.initOptim()
        self.reset_optim = Configs["reset_optim"]
        
        if Configs["cuda"]:
            self.gpu = 1
        else:
            self.gpu = 0
            
    
    def createModel(self):  
        
        model = models.__dict__[self.model_type].__dict__[self.model_name]
        return model    
    
    
    def initOptim(self):
        
        optim_args = {"params": self.model.parameters(),
                      "lr": self.lr,
                      "weight_decay": self.weight_decay}       # Use dict to contain parameters is easy to pass the torch.optim functions using **wargs
                      
        if self.optim_type in ["SGD", "RMSProp"]:
            optim_args["momentum"] = self.Configs["momentum"]
        elif self.optim_type in ['Rprop']:
            optim_args.pop["weight_decay"]
        elif self.optim_type in ['amsgrad']:
            optim_args['amsgrad'] = True
            self.Configs['optimizer'] = 'Adam'
            self.optim_type = 'Adam'
            
        self.optim = torch.optim.__dict__[self.optim_type](**optim_args)
        # Adjust learning rate
        self.lrschedule = torch.optim.lr_scheduler.MultiStepLR(self.optim, 
                                                               self.lr_schedule,
                                                               gamma=0.1)
                                                               
    
    def feedforward(self, x):
        out = self.model.forward(x)
        return out
        
    
    def loss(self, preds, targets, tasks=None):
        loss = self.loss_criterion(preds, targets)
        return loss
        
    
    def updateModel(self, inputs, targets, tasks=None):
        preds = self.forward(inputs)
        loss = self.loss(preds, targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.detach(), preds
        
        
    def learnBatch(self, dataloader):
        if self.reset_optim:
            self.initOptim()
            
        for epoch in range(self.lr_schedule[-1]):
            
            acc =  Averagemeter()
            losses = Averagemeter()
            # TODO implemeting time measurements
            
            for i, (X, Y) in enumerate(dataloader):
                
                if self.gpu:
                    X = X.cuda()
                    Y = Y.cuda()
                
                loss, preds = self.updateModel(X, Y)
                X.detach()
                Y.detach()
            
                # measure the accuracy and record the loss
                acc = self.accAccuracy(preds, Y, acc)
                losses.update(loss, X.size(0))
            
            
    def cuda(self):
        self.model = self.model.cuda()
        self.loss_criterion = self.loss_criterion.cuda()
        
        return self
        
            
    def accAccuracy(self, preds, labels, meter):
        """
        meter: accuracy measurement object
        """
        meter.update(accuracy(preds, labels), labels.shape[0]) # TODO depends on the encoding of output layer, one-hot or class label   
        return meter
        
        
    def saveModel(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():
            model_state[key] = model_state[key].cpu()
        print("==> Saving Model to: ", filename)
        torch.save(model_state, filename+".pt")
        print("==> Save Done")