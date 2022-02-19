import torch.nn as nn
import torch
import numpy as np




class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.advantage = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.reshape(-1, 16, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        return advantage

class Net2_1(nn.Module):
    def __init__(self):
        super(Net2_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.advantage = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.reshape(-1, 32, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        return advantage


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.advantage = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        return advantage



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGGMe1': [32, 128, 256, 512],
    'VGGMe2': [32, 64, 128, 256],
    'VGGMe3': [64, 128, 256, 512],
}

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features_act = self._make_layers(cfg['VGGMe1'])

        self.advantage = nn.Sequential(
            nn.Linear(512 * 16, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.reshape(-1, 32, 4, 4)
        out = self.features_act(x)
        out = out.view(out.size(0), -1)
        advantage = self.advantage(out)

        return advantage

    def _make_layers(self, cfg):
        layers = []
        in_channels = 32
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_Alpha(nn.Module):
    def __init__(self):
        super(VGG_Alpha, self).__init__()
        self.features_act = self._make_layers(cfg['VGGMe3'])

        self.probs = nn.Sequential(
            nn.Linear(512 * 16, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )

        self.value = nn.Sequential(
            nn.Linear(512 * 16, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.reshape(-1, 16, 4, 4)
        out = self.features_act(x)
        out = out.view(out.size(0), -1)
        probs = self.probs(out)

        value = self.value(out)
        return probs,value

    def _make_layers(self, cfg):
        layers = []
        in_channels = 16
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)