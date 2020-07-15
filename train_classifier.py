import os
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataloaders import get_dataloaders
from params import get_params
from classifiers import get_classifier

from tqdm import tqdm


class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset

        ###################
        # training params #
        ###################
        self.args = get_params(dataset)
        torch.manual_seed(self.args.random_seed)

        ###################
        # get dataloaders #
        ###################
        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.train_loader, self.test_loader = get_dataloaders(dataset, **kwargs)

        ######################
        # Initialize Network #
        ######################
        self.net = get_classifier(dataset)
        if self.args.cuda:
            self.net = torch.nn.DataParallel(self.net, device_ids=[0])
            self.net = self.net.cuda()

        ########################
        # Initialize Optimizer #
        ########################
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)

        #####################
        # Initialize Losses #
        #####################
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.args.n_epochs + 1)]

        ##########################
        # Checkpoint data Losses #
        ##########################
        self.curr_best = 0.0
        self.best_net_state = None
        self.best_optimizer_state = None

    def train_epoch(self, epoch):
        self.net.train()
        train_bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in train_bar:
            if self.args.cuda:
                target = target.cuda()
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                train_bar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append((batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset)))

    def test_net(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.args.cuda:
                    target = target.cuda()
                output = self.net(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        acc = 100 * correct.cpu().numpy() / len(self.test_loader.dataset)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), acc))

        if self.curr_best < acc:
            self.best_net_state = deepcopy(self.net.state_dict())
            self.best_optimizer_state = deepcopy(self.optimizer.state_dict())
            self.curr_best = acc

    def train(self):
        for epoch in range(self.args.n_epochs):
            self.train_epoch(epoch)
            self.test_net()
        output_dir = os.path.join(self.args.check_pth, str(self.curr_best))
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.best_net_state, os.path.join(output_dir, 'model.pth'))
        torch.save(self.best_optimizer_state, os.path.join(output_dir, 'optimizer.pth'))


if __name__ == '__main__':
    dataset = 'svhn'
    trainer = Trainer(dataset)
    trainer.train()






