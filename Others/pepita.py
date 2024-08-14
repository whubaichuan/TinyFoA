import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
import copy
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.autograd.function  import Function, InplaceFunction
import math


#dataset: 1-mnist,2-cifar10,3-cifar100,5-mitbih
dataset =2
#binary: 0-no binary, 1-binary
binary_w=1
binary_g=0
binary_a=1
print('binary_w: '+str(binary_w)+'binary_g: '+str(binary_g)+'binary_a: '+str(binary_a))
print('dataset: '+str(dataset))
epcoh_number=100


def MNIST_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x))])
        #Normalize((0.1307,), (0.3081,)),
        #Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=False)  # True

    test_loader = DataLoader(
        MNIST('./data', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def CIFAR10_loaders(train_batch_size=100, test_batch_size=100):

    transform = Compose([
                         transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         Lambda(lambda x: torch.flatten(x))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)
    

    return train_loader, test_loader#,val_loader

def CIFAR100_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         Lambda(lambda x: torch.flatten(x))])


    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def MITBIH_loaders():
    import mitbih_dataset.load_data as bih
    balance = 1 # 0 means no balanced (raw data), and 1 means balanced (weighted selected).
    # Please see .mithib_dataset/Distribution.png for more data structure and distribution information.
    # The above .png is from the paper-Zhang, Dengqing, et al. "An ECG heartbeat classification method based on deep convolutional neural network." Journal of Healthcare Engineering 2021 (2021): 1-9.
    x_train, y_train, x_test, y_test = bih.load_data(balance)

    # x_train = x_train[:,:169,:].reshape((x_train.shape[0],13,13))
    # x_test = x_test[:,:169,:].reshape((x_test.shape[0],13,13))
    x_train = x_train[:,:169,:].reshape((x_train.shape[0],-1))
    x_test = x_test[:,:169,:].reshape((x_test.shape[0],-1))
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_data=torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
    test_data=torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)

    return train_loader, test_loader

if dataset==1:
    input_size =784
    in_channels = 1
    output_size = 28  # Desired output size after all layers
    classes = 10
    if binary_a==1:
        learning_rate = 0.0005
    elif binary_a ==0:
        learning_rate = 0.001

    print('Dataset: mnist')
    print('learning_rate',learning_rate)
    train_loader, test_loader = MNIST_loaders()
if dataset==2:
    input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    classes = 10
    learning_rate = 0.001
    print('Dataset: cifar10')
    print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR10_loaders()
elif dataset==3:
    input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    classes = 100
    learning_rate = 0.001
    print('Dataset: cifar100')
    print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR100_loaders()
elif dataset ==5:
    input_size =math.ceil(169)
    classes = 5
    learning_rate = 0.0001
    print('Dataset: MITBIH')
    print('learning_rate',learning_rate)
    train_loader, test_loader= MITBIH_loaders()

layers = [input_size, 2000,2000,2000]

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

    def __init__(self, input_size,hidden_size,layer_index,binary_a,bias=False):
        super(BinarizeLinear, self).__init__(input_size, hidden_size,bias=False)
        self.layer_index = layer_index
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.binary_a = binary_a
    def forward(self, input):

        input_b=input

        weight_b=binarized(self.weight)

        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
            
        if self.binary_a ==1:
            return binarized(out)
        else:
            return out


class NetFC1x1024DOcust(nn.Module):
    def __init__(self,dims,classes,binary_w,binary_a):
        super().__init__()
        #self.layers = []
        if binary_w ==0:
            self.layers = torch.nn.ModuleDict(
                {f"fc{d+1}": nn.Linear(dims[d],dims[d+1],bias=False) for d in range(len(dims)-1)}
                )
        elif binary_w ==1:
            self.layers = torch.nn.ModuleDict(
                {f"fc{d+1}": BinarizeLinear(dims[d],dims[d+1],d+1,binary_a,bias=False) for d in range(len(dims)-1)}
                )
        for d in range(len(dims)-1):
            #self.layers+=[nn.Linear(dims[d],dims[d+1],bias=False)]
            nin = dims[d]
            limit = np.sqrt(6.0 / nin)
            torch.nn.init.uniform_(self.layers[f"fc{d+1}"].weight, a=-limit, b=limit)

        self.fc_last = nn.Linear(dims[-1], classes,bias=False)
        fc_last_nin = dims[-1]
        fc_last_limit = np.sqrt(6.0 / fc_last_nin)
        torch.nn.init.uniform_(self.fc_last.weight, a=-fc_last_limit, b=fc_last_limit)

    def forward(self, x, do_masks):
        x = F.relu(self.layers[f"fc{1}"](x))
        # apply dropout --> we use a custom dropout implementation because we need to present the same dropout mask in the two forward passes
        if do_masks is not None:
            i=0
            for i in range(1,len(self.layers)):
                x = x * do_masks[i-1]
                x = F.relu(self.layers[f"fc{i+1}"](x))
            x = x * do_masks[i]
            x = F.softmax(self.fc_last(x),dim=1)

        else:
            for i in range(1,len(self.layers)):
                x = F.relu(self.layers[f"fc{i+1}"](x))
            x = F.softmax(self.fc_last(x),dim=1)

        return x

# set hyperparameters
## learning rate
eta = learning_rate
## dropout keep rate
keep_rate = 0.9
## loss --> used to monitor performance, but not for parameter updates (PEPITA does not backpropagate the loss)
criterion = nn.CrossEntropyLoss()
## optimizer (choose 'SGD' o 'mom')
optim = 'mom' # --> default in the paper
if optim == 'SGD':
    gamma = 0
elif optim == 'mom':
    gamma = 0.9
## batch size
batch_size = 100 # --> default in the paper

# initialize the network
net = NetFC1x1024DOcust(layers,classes,binary_w,binary_a)


# define function to register the activations --> we need this to compare the activations in the two forward passes
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
for name, layer in net.named_modules():
    #print(name,'---',layer)
    layer.register_forward_hook(get_activation(name))


# define B --> this is the F projection matrix in the paper (here named B because F is torch.nn.functional)
nin = layers[0]
sd = np.sqrt(6/nin)
B = (torch.rand(nin,classes)*2*sd-sd)*0.05  # B is initialized with the He uniform initialization (like the forward weights)

# do one forward pass to get the activation size needed for setting up the dropout masks
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = torch.flatten(images, 1) # flatten all dimensions except batch
outputs = net(images,do_masks=None)
layers_act = []
for key in activation:
    if 'fc' in key or 'conv' in key:
        layers_act.append(F.relu(activation[key]))

# set up for momentum
if optim == 'mom':
    gamma = 0.9
    v_w_all = []
    for l_idx,w in enumerate(net.parameters()):
        #print(l_idx,'---',w.size())
        if len(w.shape)>1:
            with torch.no_grad():
                v_w_all.append(torch.zeros(w.shape))

# Train and test the model
test_accs = []
for epoch in tqdm(range(epcoh_number)):  # loop over the dataset multiple times

    # learning rate decay
    if epoch in [60,90]:
        eta = eta*0.1
        print('eta decreased to ',eta)

    # loop over batches
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, target = data
        inputs = torch.flatten(inputs, 1) # flatten all dimensions except batch
        target_onehot = F.one_hot(target,num_classes=classes)

        # create dropout mask for the two forward passes --> we need to use the same mask for the two passes
        do_masks = []
        if keep_rate < 1:
            for l in layers_act[:-1]:
                input1 = l
                do_mask = Variable(torch.ones(inputs.shape[0],input1.data.new(input1.data.size()).shape[1]).bernoulli_(keep_rate))/keep_rate
                do_masks.append(do_mask)
            do_masks.append(1) # for the last layer we don't use dropout --> just set a scalar 1 (needed for when we register activation layer)

        # forward pass 1 with original input --> keep track of activations
        outputs = net(inputs,do_masks)
        layers_act = []
        cnt_act = 0
        for key in activation:
            if 'fc' in key or 'conv' in key:
                layers_act.append(F.relu(activation[key])* do_masks[cnt_act]) # Note: we need to register the activations taking into account non-linearity and dropout mask
                #layers_act.append(F.relu(activation[key]))
                cnt_act += 1

        # compute the error
        error = outputs - target_onehot

        # modify the input with the error
        error_input = error @ B.T
        mod_inputs = inputs + error_input

        # forward pass 2 with modified input --> keep track of modulated activations
        mod_outputs = net(mod_inputs,do_masks)
        mod_layers_act = []
        cnt_act = 0
        for key in activation:
            if 'fc' in key or 'conv' in key:
                mod_layers_act.append(F.relu(activation[key])* do_masks[cnt_act]) # Note: we need to register the activations taking into account non-linearity and dropout mask
                #mod_layers_act.append(F.relu(activation[key]))
                cnt_act += 1
        mod_error = mod_outputs - target_onehot

        # compute the delta_w for the batch
        delta_w_all = []
        v_w = []
        for l_idx,w in enumerate(net.parameters()):
            v_w.append(torch.zeros(w.shape))

        for l in range(len(layers_act)):

            # update for the last layer
            if l == len(layers_act)-1:

                if len(layers_act)>1:
                    delta_w = -mod_error.T @ mod_layers_act[-2]
                else:
                    delta_w = -mod_error.T @ mod_inputs

            # update for the first layer
            elif l == 0:
                delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_inputs #.sign()

            # update for the hidden layers (not first, not last)
            elif l>0 and l<len(layers_act)-1:
                delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1] #.sign()

            delta_w_all.append(delta_w)

        # apply the weight change
        if optim == 'SGD':
            for l_idx,w in enumerate(net.parameters()):
                with torch.no_grad():
                    if binary_g ==1:
                        threshold = 0.0 
                        changed_gradient = torch.tensor(1.0)*batch_size#param.grad.data.abs().mean()
                        delta_w_all[l_idx] = torch.where(delta_w_all[l_idx]/batch_size > threshold, changed_gradient,-1*changed_gradient)
                
                    w += eta * delta_w_all[l_idx]/batch_size # specify for which layer

                    w.data = w.data.clamp_(-1,1)

        elif optim == 'mom':
            for l_idx,w in enumerate(net.parameters()):
                with torch.no_grad():
                    if binary_g ==1:
                        threshold = 0.0 
                        changed_gradient = torch.tensor(1.0)*batch_size#param.grad.data.abs().mean()
                        delta_w_all[l_idx] = torch.where(delta_w_all[l_idx]/batch_size > threshold, changed_gradient,-1*changed_gradient)

                    v_w_all[l_idx] = gamma * v_w_all[l_idx] + eta * delta_w_all[l_idx]/batch_size
                    #print(v_w_all[l_idx].shape)
                    #print(w.shape)
                    w += v_w_all[l_idx]

                    w.data = w.data.clamp_(-1,1)
                    

        # keep track of the loss
        loss = criterion(outputs, target)
        # print statistics
        running_loss += loss.item()
        if i%500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

    print('Testing...')
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            test_images = torch.flatten(test_images, 1) # flatten all dimensions except batch
            # calculate outputs by running images through the network
            test_outputs = net(test_images,do_masks=None)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(test_outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    print('Test accuracy epoch {}: {} %'.format(epoch, 100 * correct / total))
    test_accs.append(100 * correct / total)

print('Finished Training')
