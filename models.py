import torch
from torch import nn


def get_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, (7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.BatchNorm2d(32),
        nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        nn.ReLU(inplace=True),

        nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 256, (1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 512, (1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 100),
    )
