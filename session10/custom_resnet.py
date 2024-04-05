import torch
import torch.nn as nn
import torch.nn.functional as F

class X(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(X, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(3, 3),stride=1, padding=1,bias = False),
            nn.MaxPool2d(kernel_size=(2,2),stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convblock1(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),stride=1, padding=1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.conv(out)
        out = out + x
        return out

class Model_S10(nn.Module):
    def __init__(self):
        super(Model_S10, self).__init__()
        # Input Block
        self.PrepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 32

        # Layer1
        self.X1 = X(in_channels=64,out_channels=128) # output_size = 16
        self.R1 = ResBlock(in_channels=128,out_channels=128) # output_size = 32

        # Layer2
        self.X2 = X(in_channels=128,out_channels=256)

        # Layer3
        self.X3 = X(in_channels=256,out_channels=512)
        self.R3 = ResBlock(in_channels=512,out_channels=512)

        # MaxPool
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)

        # FC
        self.fc = nn.Linear(512,10)


    def forward(self, x):
        x1 = self.PrepLayer(x)

        # Layer 1
        X = self.X1(x1)
        R1 = self.R1(X)

        x1 = X + R1

        # Layer 2
        x2 = self.X2(x1)

        # Layer 3
        X = self.X3(x2)
        R2 = self.R3(X)

        x3 = X + R2

        x4 = self.maxpool(x3)

        # FC
        x5 = x4.view(x4.size(0),-1)
        x6 = self.fc(x5)

        return F.log_softmax(x6, dim=-1)