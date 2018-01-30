from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, phase2 = False):
        
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        
        for epoch in range(num_epochs):
            #TRAINING
            model.train()
            for i, (inputs, targets) in enumerate(train_loader, 1):

                inputs, targets = Variable(self.resize(inputs)), Variable(targets)
                if model.is_cuda:
                   inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                if phase2:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                optim.step()

            loss = loss.data.cpu().numpy()
            self.train_loss_history.append(loss)
            if log_nth and i % log_nth == 0:
                last_log_nth_losses = self.train_loss_history[-log_nth:]
                train_loss = np.mean(last_log_nth_losses)
                print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                    (i + epoch * iter_per_epoch,
                     iter_per_epoch * num_epochs,
                     train_loss))
            
            if phase2:
                _, preds = torch.max(outputs, 1)

                targets_mask = targets >= 0
                train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                self.train_acc_history.append(train_acc)
                if log_nth:
                    print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                       train_loss))

            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss)
                if phase2:
                    _, preds = torch.max(outputs, 1)
  
                    # Only allow images/pixels with target >= 0 e.g. for segmentation
                    scores = np.mean((preds == targets).data.cpu().numpy())
                    val_scores.append(scores)
                
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))
                
        print('FINISH.')
        
    def resize(self, data):
        
        size = data.size()
        if len(size) != 4:
            data = data.view(size[0], size[2], size[3], size[4])
        
        return data