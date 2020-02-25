import random
import torch
import torch.nn as nn

from .customLossNN import normNN

def L2(normNN):
    
    def __init__(self, learning_config):
        super(L2, self).__init__(learning_config)
        # list all the trainable parameters
        self.parameters = {n:p for n, p in self.model.named_parameters() if p.requires_grad}
        self.regularization_terms = {}
        self.task_count = 0 
        self.online_reg = True         # True: There will be only one importance matrix and previous model parameters
                                       # False: Each task has its own importance matrix and model parameters
        
    def calculateImportance(self, dataloader):
        # Use identity importance so it is an L2 regularization
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)
        return importance
        
        
    def learnBatch(self, train_loader, val_loader=None):
        
        # Learn parameters for current task
        super(L2, self).learnBatch(train_loader)
        
        # backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
            
        # calculate importance of weights for current task
        importance = self.calculateImportance(train_loader)
        
        # Sava the weight and importance and weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms) > 0:
            self.regularization_terms[1] = {"importance": importance, "task_params": task_param}
        else:
            # Use a new slot to store task-specific params, TODO tasks are mixed in batches?
            self.regularization_terms[self.task_count] = {"importance": importance, "task_params": task_param}
            
            
    def loss(self, preds, targets, tasks=None, regularization=True, **kwargs):
        loss = super(L2, self).loss(preds, targets)
        
        if regularization and len(self.regularization_terms) > 0:
            reg_loss = 0
            for i, reg_terms in self.regularization_terms.items():
               task_reg_loss = 0
               importance = reg_terms["importance"]
               task_params = reg_terms["task_params"]
            
               for n, p in self.params.items():
                   task_reg_loss += (importance[n]*(p-task_params[n])**2).sum()
               reg_loss += task_reg_loss
            loss += self.config["reg_coef"] * reg_loss
            
        return loss