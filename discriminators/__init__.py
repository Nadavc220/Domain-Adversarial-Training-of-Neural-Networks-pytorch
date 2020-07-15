from discriminators.discriminators import MnistDiscriminator

def get_discriminator(dataset):
    if dataset == 'mnist':
        return MnistDiscriminator()
