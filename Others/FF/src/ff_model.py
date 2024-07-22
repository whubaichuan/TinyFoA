import math

import torch
import torch.nn as nn

from src import utils
import numpy as np
from torch.autograd.function  import Function, InplaceFunction

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
        grad_input[input.ge(1)] = 0 #from binarynet
        grad_input[input.le(-1)] = 0 #from binarynet
        #.sign()
        return grad_input,None,None,None
        #print('bianry gradient')
        #return grad_input.sign(),None,None,None
    
    
def binarized(input,quant_mode='det'):
      return Binarize.apply(input,quant_mode)  


class BinarizeLinear(nn.Linear):

    def __init__(self, input_size,hidden_size,layer_index,binary_a):
        super(BinarizeLinear, self).__init__(input_size, hidden_size)
        self.layer_index = layer_index
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.binary_a = binary_a
    def forward(self, input):
        if self.binary_a ==1:
            if self.layer_index ==1:
                input_b=input
            else:
                input_b=binarized(input)
        else:
            input_b=input
        #input_b=input
        weight_b=binarized(self.weight)

        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        # if self.binary_a ==1:
        #     return binarized(out)
        # else:
        #     return out
        return out



class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.act_fn = ReLU_full_grad()

        input_layer_size = utils.get_input_layer_size(opt)

        # Initialize the model.
        if self.opt.input.binary_w==0:
            self.model = nn.ModuleList([nn.Linear(input_layer_size, self.num_channels[0])])
        elif self.opt.input.binary_w==1:
            self.model = nn.ModuleList([BinarizeLinear(input_layer_size, self.num_channels[0],1,self.opt.input.binary_a)])

        for i in range(1, len(self.num_channels)):
            if self.opt.input.binary_w==0:
                self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))
            elif self.opt.input.binary_w==1:
                self.model.append(BinarizeLinear(self.num_channels[i - 1], self.num_channels[i],i+1,self.opt.input.binary_a))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        # [784,2000,2000,2000]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 0)#attention: I change 1 to 0
        ) # 2000+2000+2000 = 6000
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, self.opt.input.class_num, bias=False)
        ) # 6000, 10
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    # loss incentivizing the mean activity of neurons in a layer to have low variance
    def _calc_peer_normalization_loss(self, idx, z): # z is bs*2, 2000
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0) #bsx2000 -> 2000

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        ) # the detach means that the gradient because of previous batches is not backpropagated. only the current mean activity is backpropagated
        # running_mean * 0.9 + mean_activity * 0.1

        # 2000
        # 1 = mean activation across entire layer
        #

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1) # sum of squares of each activation. bs*2

        # print("sum of squares shape: ", sum_of_squares.shape)
        # exit()
        # s - thresh    --> sigmoid --> cross entropy

        logits = sum_of_squares - z.shape[1] # if the average value of each activation is >1, logit is +ve, else -ve.
        ff_loss = self.ff_loss(logits, labels.float()) # labels are 0 or 1, so convert to float. logits->sigmoid->normal cross entropy

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels) # threshold is logits=0, so sum of squares = 784
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # print(inputs["pos_images"].shape) # bs, 1, 28, 28
        # print(inputs["neg_images"].shape) # bs, 1, 28, 28
        # print(inputs["neutral_sample"].shape) # bs, 1, 28, 28
        # print(labels["class_labels"].shape) # bs
        # exit()
        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0) # 2*bs, 1, 28, 28
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device) # 2*bs
        posneg_labels[: self.opt.input.batch_size] = 1 # first BS samples true, next BS samples false

        z = z.reshape(z.shape[0], -1) # 2*bs, 784
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        for i in range(self.opt.model.num_layers):
            scalar_outputs = self.forward_downstream_multi_pass(
                inputs, labels, scalar_outputs=scalar_outputs,index=i,
            )

        # scalar_outputs = self.forward_downstream_multi_pass(
        #     inputs, labels, scalar_outputs=scalar_outputs
        # )
        return scalar_outputs


    def forward_downstream_multi_pass(
        self, inputs, labels, scalar_outputs=None,index =-1,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        # z_all = inputs["all_sample"] # bs, num_classes, C, H, W
        # z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1) # bs, num_classes, C*H*W

        # z_all = self._layer_norm(z_all)
        # input_classification_model = []

        # with torch.no_grad():
        #     for idx, layer in enumerate(self.model):
        #         z_all = layer(z_all)
        #         z_all = self.act_fn.apply(z_all)
        #         z_unnorm = z_all.clone()
        #         z_all = self._layer_norm(z_all)

        #         if idx >= 1:
        #             # print(z.shape)
        #             input_classification_model.append(z_unnorm)

        # input_classification_model = torch.concat(input_classification_model, dim=-1) # bs x 6000 # concat all activations from all layers
        # ssq_all = torch.sum(input_classification_model ** 2, dim=-1)



        z_all = inputs["all_sample"] # bs, num_classes, C, H, W
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1) # bs, num_classes, C*H*W
        ssq_all = []
        for class_num in range(z_all.shape[1]):
            z = z_all[:, class_num, :] # bs, C*H*W
            z = self._layer_norm(z)
            input_classification_model = []

            # 784, 2000, 2000, 2000

            with torch.no_grad():
                for idx, layer in enumerate(self.model):
                    if idx < index+1:
                        z = layer(z)
                        z = self.act_fn.apply(z)
                        z_unnorm = z.clone()
                        z = self._layer_norm(z)

                        if index == 0:
                            input_classification_model.append(z_unnorm)
                        else:
                            #if idx >= 1:
                                # print(z.shape)
                            input_classification_model.append(z_unnorm)

                    # z = layer(z)
                    # z = self.act_fn.apply(z)
                    # z_unnorm = z.clone()
                    # z = self._layer_norm(z)

                    # if idx >= 1:
                    #     # print(z.shape)
                    #     input_classification_model.append(z_unnorm)

            input_classification_model = torch.concat(input_classification_model, dim=-1) # bs x 6000 # concat all activations from all layers
            ssq = torch.sum(input_classification_model ** 2, dim=-1) # bs # sum of squares of each activation
            ssq_all.append(ssq)
        ssq_all = torch.stack(ssq_all, dim=-1) # bs x num_classes # sum of squares of each activation for each class

        classification_accuracy = utils.get_accuracy(
            self.opt, ssq_all.data, labels["class_labels"]
        )

        scalar_outputs[f"multi_pass_classification_accuracy_layer{index}"] = classification_accuracy

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        # 784, 2000, 2000, 2000

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                #if idx >= 1:
                    # print(z.shape)
                input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1) # concat all activations from all layers

        # print(input_classification_model.shape)
        # exit()

        # [0.5, 1, 1.5, ....]
        # max = 3
        # [-2.5, -2, -1.5, .. 0, ..]

        output = self.linear_classifier(input_classification_model.detach()) # bs x 10 ,
        output = output - torch.max(output, dim=-1, keepdim=True)[0] # not entirely clear why each entry in output is made 0 or -ve
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs

# unclear as to why normal relu doesn't work
class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
