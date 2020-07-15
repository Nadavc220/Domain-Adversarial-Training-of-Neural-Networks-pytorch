from params.classification_params import MnistParams, SvhnParams
from params.adaptation_params import Mnist2MnistMParams


def get_params(dataset, experiment='classification'):
    if experiment == 'classification':
        if dataset == 'mnist':
            return MnistParams()
        if dataset == 'svhn':
            return SvhnParams()
    elif experiment == 'adaptation':  # adaptation experiment is defined by the source domain data
        if dataset == 'mnist':
            return Mnist2MnistMParams()