from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from utils import album_train, album_test
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
parser.add_argument('--scheduler', default='True', type=bool, help='scheduler required')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 512 if device == 'cuda' else 64 # check for GPU

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class load_dataset():
    def __init__(self):        
        self.train = datasets.CIFAR10('./Data',train=True, transform=album_train(), download=True)

        self.test = datasets.CIFAR10('./Data',train=False, transform=album_test(), download=True)

    def check_cuda(self):  
        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        # For reproducibility
        torch.manual_seed(SEED)

        if cuda:
            torch.cuda.manual_seed(SEED)
        return cuda

    def ret_datasets(self, batch_size, device):
        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if device=='cuda' else dict(shuffle=True, batch_size=64)

        # train dataloader
        train_loader = torch.utils.data.DataLoader(self.train, **dataloader_args)

        # test dataloader
        test_loader = torch.utils.data.DataLoader(self.test, **dataloader_args)
        return train_loader, test_loader


    def display_sample_imgs(self, train_loader, return_flag):
        # get some random training images
        images,labels = next(iter(train_loader))
        sample_size=25

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
            ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
            img = images[idx]/2 + 0.5
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1,2,0))
            ax.imshow(img, cmap='gray')
            ax.set_title("Label={}".format(str(classes[labels[idx]])))

        fig.tight_layout()
        plt.show()
        
        if return_flag:
            return images,labels


class train:
    # Training model
    def __init__(self):
        self.train_losses = []
        self.train_acc    = []

    def execute(self,net, device, train_loader, optimizer, scheduler, criterion,epoch):
        net.train()
        train_loss = 0
        correct = 0
        processed = 0
        pbar = tqdm(train_loader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            scheduler.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)


class test:
    # testing the model
    def __init__(self):
        self.test_losses = []
        self.test_acc    = []

    def execute(self, net, device, test_loader, criterion):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))

# load datasets for training and evaluation
dataset_obj = load_dataset() # create dataset obj
train_loader, test_loader = dataset_obj.ret_datasets(batch_size, device)


def init_training_parameters(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.003,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=24,
                                                    pct_start=0.2,
                                                    div_factor=10,
                                                    final_div_factor=25,
                                                    anneal_strategy='linear'
                                                    )
    return criterion, optimizer, scheduler