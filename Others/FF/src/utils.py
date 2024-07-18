import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import pandas as pd

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from src import ff_mnist, ff_model
import wandb


def parse_args(opt):
    #np.random.seed(opt.seed)
    #torch.manual_seed(opt.seed)
    #random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt

def get_input_layer_size(opt):
    if opt.input.dataset == "mnist":
        return 784
    elif opt.input.dataset == "senti":
        return 302
    elif opt.input.dataset == "cifar10":
        return 3072
    elif opt.input.dataset == "cifar100":
        return 3072
    elif opt.input.dataset == "mitbih":
        return 169
    else:
        raise ValueError("Unknown dataset.")

def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")


    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    # for i in model.linear_classifier.parameters():
    #     temple = i
    #     print(i)

    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer
# 784, 2000, 2000, 2000 # main params
# 6000, 10 # classification_loss params

def get_data(opt, partition):
    length_train_data = 0
    # dataset = ff_mnist.FF_MNIST(opt, partition)
    if opt.input.dataset == "mnist":
        dataset = ff_mnist.FF_MNIST(opt, partition, num_classes=10)
        length_train_data = 6000 #no use anymore
    elif opt.input.dataset == "senti":
        dataset = ff_mnist.FF_senti(opt, partition, num_classes=2)
    elif opt.input.dataset == "cifar10":
        dataset = ff_mnist.FF_CIFAR10(opt, partition, num_classes=10)
        length_train_data = 5000 #no use anymore
    elif opt.input.dataset == "cifar100":
        dataset = ff_mnist.FF_CIFAR100(opt, partition, num_classes=100)
        length_train_data = 5000 #no use anymore
    elif opt.input.dataset == "mitbih":
        dataset = ff_mnist.FF_MITBIH(opt, partition, num_classes=5)
        length_train_data = 87554 #test 21892
    else:
        raise ValueError("Unknown dataset.")

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)


    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        #worker_init_fn=seed_worker,
        #generator=g,
        num_workers=1,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_CIFAR10_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR10(
            './data',
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR10(
            './data',
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return cifar

def get_CIFAR100_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR100(
            './data',
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR100(
            './data',
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return cifar

def get_MNIST_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    if partition in ["train"]:
        mnist = torchvision.datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        mnist = torchvision.datasets.MNIST(
            './data',
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return mnist


def get_MITBIH_partition(opt, partition):

    # --------------load_data---------------------

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

    if partition in ["train"]:
        mitbih = train_data

    elif partition in ["val", "test"]:
        mitbih = test_data

    else:
        raise NotImplementedError

    return mitbih


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels

# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    #wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), "ffmnist-model.pt")
    # log model to wandb
    #wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
