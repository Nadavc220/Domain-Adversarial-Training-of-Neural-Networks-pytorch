import torch
from torch import nn


class MnistDiscriminator(nn.Module):
    """
    A classifier architecture for mnist data.
    """
    def __init__(self):
        super(MnistDiscriminator, self).__init__()
        # Encoder

        self.dense1 = nn.Linear(768, 100)
        self.dense2 = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if len(input.shape) != 2:
            input = torch.flatten(input, start_dim=1)
        x = self.dense1(input)
        x = self.relu(x)

        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
