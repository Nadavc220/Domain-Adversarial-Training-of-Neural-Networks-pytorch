import torch
from torch import nn
import torch.nn.functional as F


class MnistClassifier(nn.Module):
    """
    A classifier architecture for mnist data.
    """
    def __init__(self):
        super(MnistClassifier, self).__init__()
        # Encoder
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        # self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5)

        self.dense1 = nn.Linear(768, 100)
        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, 10)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = self.encode(input)
        x = torch.flatten(x, start_dim=1)
        x = self.decode(x)
        return x

    def encode(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        return x

    def decode(self, input):
        x = self.dense1(input)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.dense3(x)
        x = F.softmax(x, dim=1)
        return x


class SvhnClassifier(nn.Module):
    """
    A classifier architecture for mnist data.
    """
    def __init__(self):
        super(SvhnClassifier, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        # self.dense1 = nn.Linear(128, 3072)
        self.dense1 = nn.Linear(512, 3072)
        self.dense2 = nn.Linear(3072, 2048)
        self.dense3 = nn.Linear(2048, 10)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = self.encode(input)
        x = torch.flatten(x, start_dim=1)
        x = self.decode(x)
        return x

    def encode(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        return x

    def decode(self, input):
        x = self.dense1(input)
        x = self.relu(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.dense3(x)
        x = F.softmax(x, dim=1)
        return x


