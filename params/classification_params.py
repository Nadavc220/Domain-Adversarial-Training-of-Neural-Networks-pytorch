
class MnistParams:
    def __init__(self):
        self.n_epochs = 100
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.random_seed = 1
        self.cuda = True
        self.check_pth = '/home/ubuntu/nadav/GradientReversal/weights/mnist_class'


class SvhnParams:
    def __init__(self):
        self.n_epochs = 100
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.random_seed = 1
        self.cuda = True
        self.check_pth = '/home/ubuntu/nadav/GradientReversal/weights/svhn_class'