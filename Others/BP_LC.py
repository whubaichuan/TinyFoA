import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import datetime
from torch.autograd.function  import Function, InplaceFunction

start_time = datetime.datetime.now()
#dataset: 1-mnist,2-cifar10,3-cifar100,5-mitbih
dataset = 2
print('dataset: '+str(dataset))
epcoh_number=200
layer_channels = [16,32,64,64] #4
print(layer_channels)
num_layers = len(layer_channels)
kernel_size = 3
stride = 1
bias = True  # Set to True if you want bias in the layers

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

def MNIST_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([
        ToTensor()])
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
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    #trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)
    #val_loader = torch.utils.data.DataLoader(valset, batch_size=train_batch_size, shuffle=False)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)
    
    

    return train_loader, test_loader#,val_loader

def CIFAR100_loaders(train_batch_size=100, test_batch_size=100):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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

    x_train = x_train[:,:169,:].reshape((x_train.shape[0],1,13,13))
    x_test = x_test[:,:169,:].reshape((x_test.shape[0],1,13,13))
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_data=torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
    test_data=torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True,pin_memory=True,num_workers=0)

    return train_loader, test_loader

#train_loader, test_loader = CIFAR10_loaders()


if dataset==1:
    input_size =784
    in_channels = 1
    output_size = 28  # Desired output size after all layers
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: mnist')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = MNIST_loaders()
if dataset==2:
    input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    class_num = 10
    learning_rate = 0.0001
    print('Dataset: cifar10')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR10_loaders()
elif dataset==3:
    input_size =3072
    in_channels = 3
    output_size = 32  # Desired output size after all layers
    class_num = 100
    learning_rate = 0.0001
    print('Dataset: cifar100')
    #print('learning_rate',learning_rate)
    train_loader, test_loader = CIFAR100_loaders()
elif dataset ==5:
    input_size =169
    in_channels = 1
    output_size = 13
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


class LocallyConnected2d_binary(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d_binary, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            #nn.init.kaiming_uniform_(torch.empty(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2), mode='fan_in', nonlinearity='relu')
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = int((kernel_size-1)/2)
        self.pad = pad=(self.padding,self.padding,
                        self.padding,self.padding)
       
        self.dropout = nn.Dropout2d(p=0.1)
        self.bn  = nn.BatchNorm2d(out_channels)
        print('pro:'+str(self.dropout))
    def forward(self, x):
        
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = F.pad(x, self.pad, mode='constant', value=0)
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        #out = (x.unsqueeze(1) * self.weight ).sum([2, -1])
        out = (x.unsqueeze(1) * binarized(self.weight)).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        #return self.dropout(F.relu(self.bn(out)))
        #return self.dropout(F.relu(out))
        #return F.relu(self.bn(out))
        return binarized(F.relu(self.bn(out)))


class LocallyConnectedNetwork(nn.Module):
    def __init__(self, in_channels, num_layers, layer_channels, output_size, kernel_size, stride, class_num,bias=False):
        super(LocallyConnectedNetwork, self).__init__()
        self.num_layers = num_layers
        # List to hold the locally connected layers
        self.locally_connected_layers = nn.ModuleList()
        
        # Create multiple LocallyConnected2d layers
        for i in range(num_layers):
            if i == 0:
                # First layer takes input channels
                layer = LocallyConnected2d_binary(in_channels, layer_channels[i], output_size, kernel_size, stride, bias=bias)
            else:
                # Subsequent layers take previous layer's output channels
                layer = LocallyConnected2d_binary(layer_channels[i-1], layer_channels[i], output_size, kernel_size, stride, bias=bias)
            self.locally_connected_layers.append(layer)


        self.locally_connected_layers.append(nn.AdaptiveAvgPool2d(output_size=(7, 7)))
        self.locally_connected_layers.append(nn.Linear(7*7*layer_channels[-1],class_num))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Forward pass through each locally connected layer
        for layer in self.locally_connected_layers:
            if isinstance(layer, nn.Linear):
                x = self.dropout(x.view(x.shape[0],-1))
                #x = x.view(x.shape[0],-1)
                x = layer(x)
            else:
                x = layer(x)
            #print(x.shape)
        return x


model=LocallyConnectedNetwork(in_channels, num_layers, layer_channels, output_size, kernel_size, stride, class_num,bias)

print('class_num='+str(class_num))
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for name, param in model.named_parameters():
    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
    print(param.shape)

flip_tag = 0
part_numer = 2
# Train the model
total_step = len(train_loader)
model.train()
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

        for part_index in range(part_numer):
            if flip_tag%(part_numer)==part_index:
                frozen_weights_1 = model.state_dict()['locally_connected_layers.0.weight'][:,int(layer_channels[0]/part_numer)*(part_index):int(layer_channels[0]/part_numer)*(part_index+1),:,:,:,:].clone().detach()
                frozen_weights_2 = model.state_dict()['locally_connected_layers.1.weight'][:,int(layer_channels[1]/part_numer)*(part_index):int(layer_channels[1]/part_numer)*(part_index+1),int(layer_channels[0]/part_numer)*(part_index):int(layer_channels[0]/part_numer)*(part_index+1),:,:,:].clone().detach()
                frozen_weights_3 = model.state_dict()['locally_connected_layers.2.weight'][:,int(layer_channels[2]/part_numer)*(part_index):int(layer_channels[2]/part_numer)*(part_index+1),int(layer_channels[1]/part_numer)*(part_index):int(layer_channels[1]/part_numer)*(part_index+1),:,:,:].clone().detach()

        optimizer.step()

        for part_index in range(part_numer):
            if flip_tag%(part_numer)==part_index:
                model.state_dict()['locally_connected_layers.0.weight'][:,int(layer_channels[0]/part_numer)*(part_index):int(layer_channels[0]/part_numer)*(part_index+1),:,:,:,:] = frozen_weights_1.data
                model.state_dict()['locally_connected_layers.1.weight'][:,int(layer_channels[1]/part_numer)*(part_index):int(layer_channels[1]/part_numer)*(part_index+1),int(layer_channels[0]/part_numer)*(part_index):int(layer_channels[0]/part_numer)*(part_index+1),:,:,:] = frozen_weights_2.data
                model.state_dict()['locally_connected_layers.2.weight'][:,int(layer_channels[2]/part_numer)*(part_index):int(layer_channels[2]/part_numer)*(part_index+1),int(layer_channels[1]/part_numer)*(part_index):int(layer_channels[1]/part_numer)*(part_index+1),:,:,:] = frozen_weights_3.data

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 100, i+1, total_step, loss.item()))

# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
model.eval()
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

end_time = datetime.datetime.now()
total_time = (end_time-start_time).total_seconds()
print('total time: ' + str(total_time))