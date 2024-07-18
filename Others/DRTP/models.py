# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "models.py" - Construction of arbitrary network topologies.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import function
from module import FA_wrapper, TrainingHook
from torch.autograd.function  import Function, InplaceFunction
import math





class NetworkBuilder(nn.Module):
    """
    This version of the network builder assumes stride-2 pooling operations.
    """
    def __init__(self, topology, input_size, input_channels, label_features, train_batch_size, train_mode, dropout, conv_act, hidden_act, output_act, fc_zero_init, loss, device):
        super(NetworkBuilder, self).__init__()

        self.binary_w=1

        self.binary_a=1

        self.apply_softmax = (output_act == "none") and (loss == "CE")

        self.layers = nn.ModuleList()
        if (train_mode == "DFA") or (train_mode == "sDFA"):
            self.y = torch.zeros(train_batch_size, label_features, device=device)
            self.y.requires_grad = False
        else:
            self.y = None
        
        topology = topology.split('_')
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not any(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers-1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV":
                    in_channels  = input_channels if (i==0) else out_channels
                    out_channels = int(layer[1])
                    input_dim    = input_size if (i==0) else int(output_dim/2) #/2 accounts for pooling operation of the previous convolutional layer
                    output_dim   = int((input_dim - int(layer[2]) + 2*int(layer[4]))/int(layer[3]))+1
                    self.layers.append(CNN_block(
                                       in_channels=in_channels,
                                       out_channels=int(layer[1]),
                                       kernel_size=int(layer[2]),
                                       stride=int(layer[3]),
                                       padding=int(layer[4]),
                                       bias=True,
                                       activation=conv_act,
                                       dim_hook=[label_features,out_channels,output_dim,output_dim],
                                       label_features=label_features,
                                       train_mode=train_mode
                                       ))
                elif layer[0] == "FC":
                    if (i==0):
                        input_dim = pow(input_size,2)*input_channels 
                        self.conv_to_fc = 0
                    elif topology_layers[i-1][0]=="CONV":
                        input_dim = pow(int(output_dim/2),2)*int(topology_layers[i-1][1]) #/2 accounts for pooling operation of the previous convolutional layer
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim
                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers-1))
                    self.layers.append(FC_block(
                                       in_features=input_dim,
                                       out_features=output_dim,
                                       bias=True,
                                       activation=output_act if output_layer else hidden_act,
                                       dropout=dropout,
                                       dim_hook=None if output_layer else [label_features,output_dim],
                                       label_features=label_features,
                                       fc_zero_init=fc_zero_init,
                                       binary_w=self.binary_w,
                                       binary_a=self.binary_a,
                                       train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode
                                       ))
                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))

    def forward(self, x, labels):
        for i in range(len(self.layers)):
            if i == self.conv_to_fc:
                x = x.reshape(x.size(0), -1)
            x = self.layers[i](x, labels, self.y)
        
        if x.requires_grad and (self.y is not None):
            if self.apply_softmax:
                self.y.data.copy_(F.softmax(input=x.data, dim=1)) # in-place update, only happens with (s)DFA
            else:
                self.y.data.copy_(x.data) # in-place update, only happens with (s)DFA
        
        return x

class Binarize(InplaceFunction):

    def forward(ctx,input,quant_mode='det',allow_scale=True,inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()      

        scale= output.abs().mean() if allow_scale else 1 #from xnornet
        ctx.save_for_backward(input) #from binarynet
        if quant_mode=='det':
            return output.sign().mul(scale)
            #return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)

    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        #grad_input =torch.where(r.data > 1, torch.tensor(0.0), torch.tensor(1.0)) #r.detach().apply_(lambda x: 0 if x>1 else 1)
        #grad_input=grad_output
        grad_input=grad_output.clone()
        grad_input[input.ge(1)] = 0#0 #from binarynet
        grad_input[input.le(-1)] = 0#0 #from binarynet
        #.sign()
        return grad_input,None,None,None
        #print('bianry gradient')
        #return grad_input.sign(),None,None,None
    
    
def binarized(input,quant_mode='det'):
      return Binarize.apply(input,quant_mode)  


class BinarizeLinear(nn.Linear):

    def __init__(self, input_size,hidden_size,binary_a):
        super(BinarizeLinear, self).__init__(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.binary_a = binary_a
    def forward(self, input):

        input_b=input
        weight_b=binarized(self.weight)
        #weight_b=self.weight
        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        if self.binary_a ==1:
            return binarized(out)
        else:
            return out
        #return out


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, label_features, fc_zero_init, binary_w,binary_a,train_mode):
        super(FC_block, self).__init__()
        
        self.dropout = dropout
        if binary_w==0:
            self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        else:
            self.fc = BinarizeLinear(input_size=in_features, hidden_size=out_features, binary_a=binary_a)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        if self.dropout != 0:
            x = self.drop(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.hook(x, labels, y)
        return x


class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook, label_features, train_mode):
        super(CNN_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if train_mode == 'FA':
            self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape, stride=stride, padding=padding)
        self.act = Activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)

    def forward(self, x, labels, y):
        x = self.conv(x)
        x = self.act(x)
        x = self.hook(x, labels, y)
        x = self.pool(x)
        return x


class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()
        
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError("=== ERROR: activation " + str(activation) + " not supported")

    def forward(self, x):
        if self.act == None:
            return x
        else:
            return self.act(x)