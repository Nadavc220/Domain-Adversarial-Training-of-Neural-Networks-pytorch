from classifiers.classifiers import MnistClassifier, SvhnClassifier


def get_classifier(dataset):
    if dataset == 'mnist':
        return MnistClassifier()
    elif dataset == 'svhn':
        return SvhnClassifier()