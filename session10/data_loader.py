import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, PadIfNeeded, CoarseDropout,Cutout
import numpy as np
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

class album_train():
    def __init__(self):
        self.transform = Compose([
            PadIfNeeded(36,36),
            RandomCrop(32,32),
            HorizontalFlip(),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4471], always_apply=True, p=0.50),
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class album_test():
    def __init__(self):
        self.transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img 

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

    def ret_datasets(self, batch_size, cuda):
        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

        # train dataloader
        train_loader = torch.utils.data.DataLoader(self.train, **dataloader_args)

        # test dataloader
        test_loader = torch.utils.data.DataLoader(self.test, **dataloader_args)
        return train_loader, test_loader


    def display_sample_imgs(self, train_loader):
        # get some random training images
        images,labels = next(iter(train_loader))
        classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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