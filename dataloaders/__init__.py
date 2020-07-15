import torch
import torchvision
from dataloaders.mnist_m import MnistM
import numpy as np
from PIL import Image

MNIST_PATH = '/home/ubuntu/nadav/data/mnist'
SVHN_PATH = '/home/ubuntu/nadav/data/svhn'

batch_size_train = 64
batch_size_test = 1000


def get_dataloaders(data, **kwargs):
    if data == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           TripleChannel(),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size_train, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(MNIST_PATH, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           TripleChannel(),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size_test, shuffle=True, **kwargs)
        return train_loader, test_loader

    elif data == 'svhn':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(SVHN_PATH, split='train', download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.4376821182313768, 0.4437697182528218, 0.47280443574441605),
                                               (0.19803012860897073, 0.2010156285476501, 0.19703614495525437))
                                       ])),
            batch_size=batch_size_train, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(SVHN_PATH, split='test', download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.4376821182313768, 0.4437697182528218, 0.47280443574441605),
                                               (0.19803012860897073, 0.2010156285476501, 0.19703614495525437))
                                       ])),
            batch_size=batch_size_test, shuffle=True, **kwargs)
        return train_loader, test_loader

    elif data == 'mnist_m':
        train_loader = torch.utils.data.DataLoader(MnistM(split='train'),
                                              batch_size=batch_size_train, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(MnistM(split='test'),
                                             batch_size=batch_size_test, shuffle=True, **kwargs)
        return train_loader, test_loader


class TripleChannel(object):

    def __call__(self, sample):
        x = np.array(sample)[np.newaxis]
        x = np.tile(x, (3, 1, 1))
        x = Image.fromarray(x.transpose(1, 2, 0))

        return x


