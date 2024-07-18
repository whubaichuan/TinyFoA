import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd.function  import Function, InplaceFunction
import math

   
#dataset: 1-mnist,2-cifar10,3-cifar100,5-mitbih
dataset = 1
#binary: 0-no binary, 1-binary
binary_w=1
binary_g=0
binary_a=1
print('binary_w: '+str(binary_w)+'binary_g: '+str(binary_g)+'binary_a: '+str(binary_a))
print('dataset: '+str(dataset))
epcoh_number=100

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

def MNIST_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x))])
        #Normalize((0.1307,), (0.3081,)),
        #Lambda(lambda x: torch.flatten(x))])

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

    transform = Compose([
                         transforms.ToTensor(),
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
    

    return train_loader, test_loader#,val_loader

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
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: mnist')
    print('learning_rate',learning_rate)
    train_loader, test_loader = MNIST_loaders()
if dataset==2:
    input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: cifar10')
    print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR10_loaders()
elif dataset==3:
    input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    class_num = 100
    learning_rate = 0.0001
    print('Dataset: cifar100')
    print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR100_loaders()
elif dataset ==5:
    input_size =math.ceil(169)
    class_num = 5
    learning_rate = 0.0001
    print('Dataset: MITBIH')
    print('learning_rate',learning_rate)
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

        weight_b=binarized(self.weight)

        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class Net(nn.Module):
    def __init__(self,layer_number,input_size,class_num,binary_w,binary_a):
        super(Net, self).__init__()
        if binary_w ==0:
            self.conv1 = nn.Linear(input_size,2000)
            self.conv2 = nn.Linear(2000,2000)
            self.conv3 = nn.Linear(2000,2000)
            self.conv4 = nn.Linear(2000,2000)
        elif binary_w ==1:
            self.conv1 = BinarizeLinear(input_size,2000,1,binary_a)
            self.conv2 = BinarizeLinear(2000,2000,2,binary_a)
            self.conv3 = BinarizeLinear(2000,2000,3,binary_a)
            self.conv4 = BinarizeLinear(2000,2000,4,binary_a)

        self.fc1 = nn.Linear(2000, class_num)
        self.layer_number = layer_number+1

    def forward(self, x):
        if self.layer_number > 0:
            x = self.conv1(x)
            x = F.relu(x)
        if self.layer_number >1:
            x = self.conv2(x)
            x = F.relu(x)
        if self.layer_number >2:
            x = self.conv3(x)
            x = F.relu(x)
        if self.layer_number>3:
            x = self.conv4(x)
            x = F.relu(x)

        output = self.fc1(x)
        return output
    

# create a validation set
for layer_number in range(3,4):
    model=Net(layer_number,input_size,class_num,binary_w,binary_a)
    print(model)
    model.to(device)
    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    flip_tag = 0
    part_numer = 2
    # Train the model
    total_step = len(train_loader)
    for epoch in range(epcoh_number):
        flip_tag +=1

        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()

            for layer_index in range(layer_number):
                if layer_index == 0:
                    input_length = input_size/part_numer
                else:
                    input_length = 2000/part_numer

                output_lenth = 2000/part_numer

            for part_index in range(part_numer):
                if flip_tag%(part_numer)==part_index:

                    frozen_weights_1 = model.state_dict()['conv1.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_size/part_numer)*part_index:int(input_size/part_numer)*(part_index+1)].clone().detach()
                    frozen_weights_2 = model.state_dict()['conv2.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_length)*part_index:int(input_length)*(part_index+1)].clone().detach()
                    frozen_weights_3 = model.state_dict()['conv3.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_length)*part_index:int(input_length)*(part_index+1)].clone().detach()
                    frozen_weights_4 = model.state_dict()['conv4.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_length)*part_index:int(input_length)*(part_index+1)].clone().detach()


            optimizer.step()

            if binary_w ==1:
                for param in model.parameters():
                    param.data = param.data.clamp_(-1,1)

            for part_index in range(part_numer):
                if flip_tag%(part_numer)==part_index:
                    model.state_dict()['conv1.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_size/part_numer)*part_index:int(input_size/part_numer)*(part_index+1)] = frozen_weights_1.data
                    model.state_dict()['conv2.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_length)*part_index:int(input_length)*(part_index+1)] = frozen_weights_2.data
                    model.state_dict()['conv3.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_length)*part_index:int(input_length)*(part_index+1)] = frozen_weights_3.data
                    model.state_dict()['conv4.weight'][int(output_lenth)*part_index:int(output_lenth)*(part_index+1),int(input_length)*part_index:int(input_length)*(part_index+1)] = frozen_weights_4.data


            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, 100, i+1, total_step, loss.item()))

    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
