import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision
import math
from torch.autograd.function  import Function, InplaceFunction


#dataset: 1-mnist,2-cifar10,3-cifar100, 5-mitbih
dataset = 1
scale = 1
layers = 4
length = int(2000/scale)
#initialization:1-fixed normal distributon for each layer,2-normal initialization he,3-random projection
initialization = 2
epcoh_number=100

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

# load data
def MNIST_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=False)  # True

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def CIFAR10_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Lambda(lambda x: torch.flatten(x))])

    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def CIFAR100_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Lambda(lambda x: torch.flatten(x))])

    trainset = torchvision.datasets.CIFAR100(root='./data/', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR100(root='./data/', train=False,
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

    x_train = x_train[:,:169,:].reshape((x_train.shape[0],13,13))
    x_test = x_test[:,:169,:].reshape((x_test.shape[0],13,13))
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_data=torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
    test_data=torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)

    return train_loader, test_loader


if dataset==1:
    input_size =math.ceil(784/scale)
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: mnist')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = MNIST_loaders()
elif dataset==2:
    input_size =math.ceil(3072/scale)
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: cifar10')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR10_loaders()
elif dataset==3:
    input_size =math.ceil(3072/scale)
    class_num = 100
    learning_rate = 0.0001
    print('Dataset: cifar100')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR100_loaders()
elif dataset ==5:
    input_size =math.ceil(169/scale)
    class_num = 5
    learning_rate = 0.0001
    print('Dataset: MITBIH')
    #print('learning_rate',learning_rate)
    train_loader, test_loader= MITBIH_loaders()



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
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)

    def backward(ctx,grad_output):
        #STE 

        input, = ctx.saved_tensors
        #grad_input =torch.where(r.data > 1, torch.tensor(0.0), torch.tensor(1.0)) #r.detach().apply_(lambda x: 0 if x>1 else 1)
        #grad_input=grad_output
        grad_input=grad_output.clone()
        grad_input[input.ge(1)] = 0 #from binarynet
        grad_input[input.le(-1)] = 0 #from binarynet
        #.sign()
        
        return grad_input,None,None,None
    
    
def binarized(input,quant_mode='det'):
      return Binarize.apply(input,quant_mode)  


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):


        #input_b=binarized(input)
        input_b=input
        weight_b=binarized(self.weight)
        #weight_b=self.weight
        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        #return out
        return binarized(out)


class Net_one(nn.Module):
    def __init__(self,length,input_size,class_num):
        super(Net_one, self).__init__()
        # self.fc = nn.Linear(input_size,length)
        # self.lastfc = nn.Linear(length, class_num)
        self.fc = BinarizeLinear(input_size,length)
        self.lastfc = nn.Linear(length, class_num)

    def forward(self, x):
        x = self.fc(x[:,::scale])
        x_internal = F.relu(x)
        x = self.lastfc(x_internal)
        output = x
        return output,x_internal

class Net_more(nn.Module):
    def __init__(self,model,input_features, out_features,class_num):
        super(Net_more, self).__init__()
        self.previous_model=model
        # self.fc = nn.Linear(input_size,length)
        # self.lastfc = nn.Linear(length, class_num)
        self.fc = BinarizeLinear(length,length)
        self.lastfc = nn.Linear(length, class_num)

    def forward(self, x):
        _,x = self.previous_model(x)
        x_internal = F.relu(self.fc(x))
        x = self.lastfc(x_internal)
        output = x
        return output,x_internal


for layer_index in range(layers):
    if layer_index == 0:
        new_model=Net_one(length,input_size,class_num).to(device)

    else:
        for param in new_model.parameters():
            param.requires_grad = False
        new_model = Net_more(new_model,length,length,class_num).to(device)


    for name, param in new_model.named_parameters():
        if name =='lastfc.weight' or name=='lastfc.bias':
            param.requires_grad = False

            
    if initialization==1:
        fixed_weights = torch.randn((class_num,length))
        new_model.lastfc.weight.data = fixed_weights
        print('initialization:fixed normal distributon for each layer')
    elif initialization==2:
        #torch.nn.init.kaiming_uniform_(new_model.lastfc.weight)
        torch.nn.init.kaiming_normal_(new_model.lastfc.weight)
        print('initialization:normal initialization he')
    elif initialization==3:
        matrix = np.zeros((class_num,length))
        total_elements = length*class_num
        one_third_elements = total_elements // 6
        indices_to_set_to_one = np.random.choice(total_elements, one_third_elements, replace=False)
        indices_to_set_to_negone = np.random.choice(np.delete(np.arange(total_elements),indices_to_set_to_one), one_third_elements, replace=False)
        matrix.flat[indices_to_set_to_one] = 1
        matrix.flat[indices_to_set_to_negone] = -1
        fixed_weights = torch.tensor(matrix).to(dtype=torch.float32)
        new_model.lastfc.weight.data = fixed_weights
        print('initialization:random projection')

    
    new_model.lastfc.bias.data = torch.zeros((class_num)).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)

    flip_tag = 0
    part_numer = 2
    # Train the model
    total_step = len(train_loader)
    for epoch in range(epcoh_number):
        flip_tag +=1

        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device) if dataset<4 else images.reshape(images.shape[0],-1).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs,_= new_model(images)
            loss = criterion(outputs, labels)

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()

            if layer_index == 0:
                input_length = input_size/part_numer
            else:
                input_length = length/part_numer

            output_lenth = length/part_numer
            
            for part_index in range(part_numer):
                if flip_tag%(part_numer)==part_index:
                    frozen_weights = new_model.state_dict()['fc.weight'][int(output_lenth)*(part_index):int(output_lenth)*(part_index+1),int(input_length)*(part_index):int(input_length)*(part_index+1)].clone().detach()

            optimizer.step()


            for part_index in range(part_numer):
                if flip_tag%(part_numer)==part_index:
                    new_model.state_dict()['fc.weight'][int(output_lenth)*(part_index):int(output_lenth)*(part_index+1),int(input_length)*(part_index):int(input_length)*(part_index+1)] = frozen_weights.data

            #from binaryconnect
            #clamp to -1 to 1
            for param in new_model.parameters():
               param.data = param.data.clamp_(-1,1)


            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 100, i+1, total_step, loss.item()))

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device) if dataset<4 else images.reshape(images.shape[0],-1).to(device)
            labels = labels.to(device)
            outputs,_ = new_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network in {layer_index}th layer on the {total} test images: {100 * correct / total}" )


