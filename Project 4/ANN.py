'''
ANN.py
------
Contains class ANN(torch.nn.Module)
-----------------------------------
ANN is a simple neural network, created with the intent of performing multi class classification
'''

from turtle import forward
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import os
from tqdm import tqdm as pbar

def training_loop(net, train_data, train_labels, test_data, test_labels, loss_criterion, loss_margin=0.1):
    net = net.train()  # set training mode
    lrn_rate = 1e-4
    bat_size = 125
    loss_func = loss_criterion.cuda() 
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    max_epochs = 40000
    n_items = len(train_data)
    batcher = Batcher(n_items, bat_size)
    
    losses = []
    train_acc = []
    test_acc = []
    for epoch in pbar(range(max_epochs)):
        loss = 0.
        if epoch > 0 and epoch % (10) == 0:
            print('epoch = %6d' % epoch, end='')
            print('  batch loss = %7.4f' % np.average(losses[-10:]), end='')
            acc = ANN.akkuracy(net, train_data, train_labels)
            print(' train accuracy = %0.2f' % acc, end='')
            train_acc.append(acc)
            acc = ANN.akkuracy(net, test_data, test_labels)
            print(' dev accuracy = %0.2f' % acc)
            test_acc.append(acc)
        for curr_bat in batcher:
            X1 = T.Tensor(train_data[curr_bat]).cuda()
            if(net.dataset_type == 'binary'):
                Y1 = T.Tensor(train_labels[curr_bat]).cuda()
            else:
                Y1 = T.LongTensor(train_labels[curr_bat]).cuda()
            optimizer.zero_grad()
            oupt = net(X1)
            oupt = T.squeeze(oupt)
            loss_obj = loss_func(oupt, Y1)
            loss_obj.backward()
            l = loss_obj.item()
            loss += l
            optimizer.step()
        losses.append(loss)
        if(loss < loss_margin):
            break
    return losses
class Batcher:
    def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  # exit calling for-loop
        else:
            result = self.indices[self.ptr:self.ptr+self.batch_size]
            self.ptr += self.batch_size
            return result
class ANN(T.nn.Module):
    def __init__(self, input_dim, dataset_type='binary',args=None):
        super(ANN, self).__init__()
        if(dataset_type == None):
            print("Dataset type not mentioned. Assuming Synthetic dataset.")
            dataset_type = 'binary'
        self.dataset_type = dataset_type
        if(dataset_type == 'multi_class'):
            if(args==None):
                input = input_dim
                hid1 = 23*30
                hid2 = 23*9
                hid3 = 150
                hid4 = 25
            else:
                input,hid1,hid2,hid3,hid4 = args
            self.hid1 = T.nn.Linear(input, hid1) 
            self.hid2 = T.nn.Linear(hid1, hid2)
            self.hid3 = T.nn.Linear(hid2, hid3)
            self.hid4 = T.nn.Linear(hid3, hid4)
            self.oupt = T.nn.Linear(hid4, 5)

            T.nn.init.xavier_uniform_(self.hid1.weight)
            T.nn.init.zeros_(self.hid1.bias)
            T.nn.init.xavier_uniform_(self.hid2.weight)
            T.nn.init.zeros_(self.hid2.bias)
            T.nn.init.xavier_uniform_(self.hid3.weight)
            T.nn.init.zeros_(self.hid3.bias)
            T.nn.init.xavier_uniform_(self.hid4.weight)
            T.nn.init.zeros_(self.hid4.bias)
            T.nn.init.xavier_uniform_(self.oupt.weight)
            T.nn.init.zeros_(self.oupt.bias)
            pass
        else:
            if(args==None):
                hid1 = 10
                hid2 = 5
            else:
                hid1,hid2 = args
            self.hid1 = T.nn.Linear(input_dim, hid1)
            self.hid2 = T.nn.Linear(hid1, hid2)
            self.oupt = T.nn.Linear(hid2, 1)

            T.nn.init.xavier_uniform_(self.hid1.weight)
            T.nn.init.zeros_(self.hid1.bias)
            T.nn.init.xavier_uniform_(self.hid2.weight)
            T.nn.init.zeros_(self.hid2.bias)
            T.nn.init.xavier_uniform_(self.oupt.weight)
            T.nn.init.zeros_(self.oupt.bias)
            pass
        self.history = None
        
    # def __call__(self,X,Y,n_epochs=100,bat_size=10):
    #     self.history = self.classifier.fit(X,Y,batch_size=bat_size,epochs=n_epochs,verbose=1) 
    #     return self.history
    
    def forward(self, x):
        if(self.dataset_type == 'binary'):
            z = T.tanh(self.hid1(x)) 
            z = T.tanh(self.hid2(z))
            z = T.sigmoid(self.oupt(z))
            return z
        else:
            z = T.tanh(self.hid1(x)) 
            z = T.tanh(self.hid2(z))
            z = T.tanh(self.hid3(z))
            z = T.tanh(self.hid4(z))
            z = T.sigmoid(self.oupt(z))
            return z
        
    def akkuracy(model, data_x, data_y):
        if(model.dataset_type == 'binary'):
            return ANN.akkuracy_bin(model, data_x, data_y)
        else:
            return ANN.akkuracy_mul(model, data_x, data_y)
        
    def akkuracy_bin(model, data_x, data_y):
        # data_x and data_y are numpy array-of-arrays matrices
        X2 = T.Tensor(data_x).cuda()
        Y2 = T.ByteTensor(data_y).cuda()   # a Tensor of 0s and 1s
        oupt = model(X2)            # a Tensor of floats
        pred_y = (oupt >= 0.5).squeeze()       # a Tensor of 0s and 1s
        num_correct = T.sum(Y2==pred_y)  # a Tensor
        acc = (num_correct.item() / len(data_y))  # scalar
        return acc
    
    def akkuracy_mul(model, data_x, data_y):
        # data_x and data_y are numpy array-of-arrays matrices
        X2 = T.Tensor(data_x).cuda()
        Y2 = T.LongTensor(data_y).cuda()   
        oupt = model(X2)            # a Tensor of floats
        pred_y = (T.argmax(oupt,dim=-1)).squeeze()
        num_correct = T.sum(Y2==pred_y)  # a Tensor
        acc = (num_correct.item() / len(data_y))  # scalar
        return acc
    
     