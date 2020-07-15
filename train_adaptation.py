import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

from dataloaders import get_dataloaders
from classifiers import get_classifier
from discriminators import get_discriminator
from params import get_params

class GradReverse(Function):
    lambd = 0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * GradReverse.lambd


class DomainClassifier(nn.Module):
    """
    A wrapper for the run of encoder-rg-discriminator in order to run net in one
    back-propagation as described in paper.
    """
    def __init__(self, encoder, discriminator):
        super(DomainClassifier, self).__init__()
        self.encoder = encoder
        self.discriminator = discriminator
        self.lambd = 0

    def update_lambd(self, lambd):
        self.lambd = lambd
        GradReverse.lambd = self.lambd

    def forward(self, input):
        x = self.encoder(input)
        x = GradReverse.apply(x)
        x = self.discriminator(x)
        return x


class GRDomainAdaptation:

    def __init__(self, source_dataset):
        ###########################
        # Initialize Info Holders #
        ###########################
        self.args = get_params(source_dataset, experiment='adaptation')
        self.source_best_pred = 0.0
        self.target_best_pred = 0.0
        self.best_source_net_state = None
        self.best_target_net_state = None
        self.source_test_losses = []
        self.target_test_losses = []
        self.source_test_acc = []
        self.target_test_acc = []
        self.iters = 0

        #######################################
        # Initialize Source and target labels #
        #######################################
        self.source_disc_labels = torch.zeros(size=(self.args.batch_size_train, 1)).requires_grad_(False)
        self.target_disc_labels = torch.ones(size=(self.args.batch_size_train, 1)).requires_grad_(False)
        if self.args.cuda:
            self.source_disc_labels = self.source_disc_labels.cuda()
            self.target_disc_labels = self.target_disc_labels.cuda()

        ######################
        # Define DataLoaders #
        ######################
        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.source_train_loader, self.source_test_loader = get_dataloaders(self.args.source_dataset, **kwargs)
        self.target_train_loader, self.target_test_loader = get_dataloaders(self.args.target_dataset, **kwargs)
        self.n_batch = min(len(self.target_train_loader), len(self.source_train_loader))

        ##################
        # Define network #
        ##################
        self.net = get_classifier(source_dataset)

        if self.args.cuda:
            self.net = torch.nn.DataParallel(self.net, device_ids=[0])
            self.net = self.net.cuda()

        ###############
        # Set Encoder #
        ###############
        if self.args.cuda:
            self.encoder = self.net.module.encode
        else:
            self.encoder = self.net.encode

        ###################################################
        # Set Domain Classifier (Encoder + Discriminator) #
        ###################################################
        self.discriminator = get_discriminator(source_dataset)
        self.domain_classifier = DomainClassifier(self.encoder, self.discriminator)
        if self.args.cuda:
            self.domain_classifier = torch.nn.DataParallel(self.domain_classifier, device_ids=[0])
            self.domain_classifier = self.domain_classifier.cuda()

        #####################
        # Define Optimizers #
        #####################
        self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)
        self.encoder_optimizer = torch.optim.SGD(self.net.parameters(), self.args.learning_rate, momentum=self.args.momentum)
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)
        # self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))

        # Define lr scheduler

        # self.disc_criterion = torch.nn.BCELoss()

    def train_epoch(self):
        self.net.train()
        tbar = tqdm(enumerate(zip(self.source_train_loader, self.target_train_loader)))
        net_loss = 0.0
        disc_loss = 0.0
        total_loss = 0.0

        for i, ((source_img, source_label), (target_img, _)) in tbar:
            ##############################
            # update learning parameters #
            ##############################
            self.iters += 1
            p = self.iters / (self.args.n_epochs * self.n_batch)

            lambd = (2. / (1. + np.exp(-10. * p))) - 1
            if self.args.cuda:
                self.domain_classifier.module.update_lambd(lambd)
            else:
                self.domain_classifier.update_lambd(lambd)

            lr = self.args.learning_rate / (1. + 10 * p) ** 0.75
            self.discriminator_optimizer.lr = lr
            self.net_optimizer.lr = lr

            #########################################################################
            # set batch size in cases where source and target domain differ in size #
            #########################################################################
            curr_batch_size = min(source_img.shape[0], target_img.shape[0])
            source_img = source_img[:curr_batch_size]
            source_label = source_label[:curr_batch_size]
            target_img = target_img[:curr_batch_size]
            source_disc_labels = self.source_disc_labels[:curr_batch_size]
            target_disc_labels = self.target_disc_labels[:curr_batch_size]
            if self.args.cuda:
                source_img, source_label = source_img.cuda(), source_label.cuda()
                target_img = target_img.cuda()

            #######################################################
            # Train network (Encoder + Classifier) on Source Data #
            #######################################################
            self.net_optimizer.zero_grad()
            net_output = self.net(source_img)
            class_net_loss = F.nll_loss(net_output, source_label)
            class_net_loss.backward()
            self.net_optimizer.step()
            net_loss += class_net_loss

            #########################################
            # Train encoder on Source + Target data #
            #########################################
            self.encoder_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            dom_input = torch.cat([source_img, target_img], dim=0)
            dom_labels = torch.cat([source_disc_labels, target_disc_labels], dim=0)
            dom_output = self.domain_classifier(dom_input)
            dom_loss = F.binary_cross_entropy(dom_output, dom_labels)
            # source_disc_output = self.domain_classifier(source_img)
            # target_disc_output = self.domain_classifier(target_img)
            # source_loss = F.binary_cross_entropy(source_disc_output, source_disc_labels)
            # target_loss = F.binary_cross_entropy(target_disc_output, target_disc_labels)

            # calculate total loss value
            # domain_class_loss = source_loss + target_loss  # TODO: /2?
            dom_loss.backward()
            self.discriminator_optimizer.step()
            self.encoder_optimizer.step()
            disc_loss += dom_loss
            # disc_loss += domain_class_loss / 2  # TODO /2?

            total_loss += class_net_loss - lambd * dom_loss

            tbar.set_description('Net loss: {0:.6f}; Discriminator loss: {1:.6f}; Total Loss: {2:.6f}; {3:.2f}%;'.format((net_loss / (i + 1)),
                                                                                                                         (disc_loss / (i + 1)),
                                                                                                                         (total_loss / (i + 1)),
                                                                                                                         (i + 1) / self.n_batch * 100))

    def test_net(self):
        ####################
        # Test Source Data #
        ####################
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in self.source_test_loader:
                if self.args.cuda:
                    labels = labels.cuda()
                output = self.net(data)
                test_loss += F.nll_loss(output, labels, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()
        test_loss /= len(self.source_test_loader.dataset)
        acc = 100 * correct.cpu().numpy() / len(self.source_test_loader.dataset)
        print('Source Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
            test_loss, correct, len(self.source_test_loader.dataset), acc))

        self.source_test_losses.append(test_loss)
        self.source_test_acc.append(acc)

        if self.source_best_pred < acc:
            self.best_source_net_state = deepcopy(self.net.state_dict())
            self.source_best_pred = acc
            # self.best_optimizer_state = deepcopy(self.optimizer.state_dict())

        ####################
        # Test Target Data #
        ####################
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in self.target_test_loader:
                if self.args.cuda:
                    labels = labels.cuda()
                output = self.net(data)
                test_loss += F.nll_loss(output, labels, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()
        test_loss /= len(self.target_test_loader.dataset)
        acc = 100 * correct.cpu().numpy() / len(self.target_test_loader.dataset)
        print('Target Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
            test_loss, correct, len(self.target_test_loader.dataset), acc))

        self.target_test_losses.append(test_loss)
        self.target_test_acc.append(acc)

        if self.target_best_pred < acc:
            self.best_target_net_state = deepcopy(self.net.state_dict())
            self.target_best_pred = acc

    def plot_acc_info(self, output_dir):
        t = np.arange(1, len(self.source_test_acc) + 1, 1)

        fig, ax = plt.subplots()
        ax.plot(t, self.source_test_acc, label='Source')
        ax.plot(t, self.target_test_acc, label='Target')

        ax.set(xlabel='Epoch', ylabel='Acc', title='Source vs. Target Test Accuracies')
        ax.grid()
        plt.legend()

        file_name = 'accuracies.png'

        path = os.path.join(output_dir, file_name)
        fig.savefig(path)

    def train(self):
        for epoch in range(self.args.n_epochs):
            print('Epoch: {}; Source Best: {}; Target Best: {}'.format(epoch, self.source_best_pred, self.target_best_pred))
            self.train_epoch()
            self.test_net()
        output_dir = os.path.join(self.args.check_pth, str(self.target_best_pred))
        os.makedirs(output_dir, exist_ok=True)
        self.plot_acc_info(output_dir)
        torch.save(self.best_source_net_state, os.path.join(output_dir, 'source_model.pth'))
        torch.save(self.best_target_net_state, os.path.join(output_dir, 'target_model.pth'))


if __name__ == '__main__':
    source_dataset = 'mnist'
    trainer = GRDomainAdaptation(source_dataset)
    trainer.train()




# for i in range(num_epochs):
#     source_gen = batch_gen(source_batches, source_idx, Xs_train, ys_train)
#     target_gen = batch_gen(target_batches, target_idx, Xt_train, None)
#
#     # iterate over batches
#     for (source_sample, source_label) in source_gen:
#
#         # update lambda and learning rate as suggested in the paper
#         p = float(j) / num_steps
#         lambd = 2. / (1. + np.exp(-10. * p)) - 1
#         lr = 0.01 / (1. + 10 * p) ** 0.75
#         d_optimizer.lr = lr
#         c_optimizer.lr = lr
#         f_optimizer.lr = lr
#
#         # exit if batch size incorrect, get next target batch
#         if len(source_sample) != batch_size / 2:
#             continue
#         target_sample = next(target_gen)
#
#         # concatenate source and target batch
#         concat_sample = torch.cat([source_sample, target_sample], 0)
#
#         # 1) train feature_extractor and class_classifier on source batch
#         # reset gradients
#         f_ext.zero_grad()
#         c_clf.zero_grad()
#
#         # calculate class_classifier predictions on source samples
#         c_out = c_clf(f_ext(source_sample).view(batch_size // 2, -1))
#
#         # optimize feature_extractor and class_classifier on output
#         f_c_loss = (c_out, source_label.float())
#         f_c_loss.backward(retain_variables=True)
#         c_optimizer.step()
#         f_optimizer.step()
#
#         # 2) train feature_extractor and domain_classifier on full batch x
#         # reset gradients
#         f_ext.zero_grad()
#         d_clf.zero_grad()
#
#         # calculate domain_classifier predictions on batch x
#         d_out = d_clf(f_ext(concat_sample).view(batch_size, -1))
#
#         # optimize feature_extractor and domain_classifier with output
#         f_d_loss = d_crit(d_out, yd.float())
#         f_d_loss.backward(retain_variables=True)
#         d_optimizer.step()
#         f_optimizer.step()
