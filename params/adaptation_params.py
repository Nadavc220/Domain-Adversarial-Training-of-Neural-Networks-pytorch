
class Mnist2MnistMParams:
    def __init__(self):
        self.source_dataset = 'mnist'
        self.target_dataset = 'mnist_m'
        self.n_epochs = 50
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.log_interval = 10

        self.random_seed = 1
        self.cuda = True
        self.check_pth = '/home/ubuntu/nadav/GradientReversal/weights/mnist2mnist_m'