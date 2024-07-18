import numpy as np
import torch

from src import utils

class FF_CIFAR100(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=100):
        self.opt = opt
        self.mnist = utils.get_CIFAR100_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        #pos_sample[0, 0, : self.num_classes] = one_hot_label
        pos_sample.view(1,-1)[0, : self.num_classes] = one_hot_label*self.opt.input.label_weight
        #pos_sample[0, 0, : ] = one_hot_label[0:32]
        #pos_sample[0, 1, : ] = one_hot_label[32:64]
        #pos_sample[0, 2, : ] = one_hot_label[64:96]
        #pos_sample[0, 3, : 4] = one_hot_label[96:]
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        #neg_sample[0, 0, : self.num_classes] = one_hot_label
        neg_sample.view(1,-1)[0, : self.num_classes] = one_hot_label*self.opt.input.label_weight
        #neg_sample[0, 0, : ] = one_hot_label[0:32]
        #neg_sample[0, 1, : ] = one_hot_label[32:64]
        #neg_sample[0, 2, : ] = one_hot_label[64:96]
        #neg_sample[0, 3, : 4] = one_hot_label[96:]
        return neg_sample

    def _get_neutral_sample(self, z):
        #z[0, 0, : self.num_classes] = self.uniform_label
        z.view(1,-1)[0, : self.num_classes] = self.uniform_label
        #z[0, 0, : ] = self.uniform_label[0:32]
        #z[0, 1, : ] = self.uniform_label[32:64]
        #z[0, 2, : ] = self.uniform_label[64:96]
        #z[0, 3, : 4] = self.uniform_label[96:]
        return z

    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            #all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()
            one_hot_label_new = one_hot_label.clone()
            all_samples[i].view(1,-1)[0, : self.num_classes] = one_hot_label_new*self.opt.input.label_weight
            #all_samples[i,0, 0, : ] = one_hot_label_new[0:32]
            #all_samples[i,0, 1, : ] = one_hot_label_new[32:64]
            #all_samples[i,0, 2, : ] = one_hot_label_new[64:96]
            #all_samples[i,0, 3, : 4] = one_hot_label_new[96:]
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label

class FF_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_CIFAR10_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, : self.num_classes] = one_hot_label*self.opt.input.label_weight
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label*self.opt.input.label_weight
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z

    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()*self.opt.input.label_weight
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label

class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, : self.num_classes] = one_hot_label*self.opt.input.label_weight
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label*self.opt.input.label_weight
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z

    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()*self.opt.input.label_weight
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label



class FF_MITBIH(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=5):
        self.opt = opt
        self.mnist = utils.get_MITBIH_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, : self.num_classes] = one_hot_label*self.opt.input.label_weight#*20 #caution: need to change based on the dataset
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label*self.opt.input.label_weight#*20 #caution: need to change based on the dataset
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z

    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()*self.opt.input.label_weight#*20 #caution: need to change based on the dataset
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        sample = sample.unsqueeze(0)
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label