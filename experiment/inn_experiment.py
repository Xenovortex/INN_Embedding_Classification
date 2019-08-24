from time import time
import torch
import numpy as np
from functionalities import dataloader as dl
from functionalities import filemanager as fm
from functionalities import plot as pl
from architecture import generative_classifier as gc
from architecture import model as m
from tqdm import tqdm_notebook as tqdm


class inn_experiment:
    """
    Class for training INN models
    """

    def __init__(self, num_epoch, batch_size, milestones, modelname, device='cpu', lr_init=5e-4, mu_init=11., beta=5.0,
                 interval_log=1, interval_checkpoint=5, interval_figure=20):
        """
        Init class with pretraining setup.

        :param num_epoch: number of training epochs
        :param batch_size: Batch Size to use during training.
        :param lr_init: Starting learning rate. Will be decrease with adaptive learning.
        :param milestones: list of training epochs at which the learning rate will be reduce
        :param get_model: model for training
        :param modelname: model name under which the model will be saved
        :param device: device on which to do the computation (CPU or CUDA). Please use get_device() to get device
        variable, if using multiple GPU's.
        :param weight_decay: weight decay (L2 penalty) for adam optimizer
               """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.modelname = modelname
        self.device = device
        self.beta = beta
        self.interval_log = interval_log
        self.interval_checkpoint = interval_checkpoint
        self.interval_figure = interval_figure

        self.vgg = m.get_vgg16().to(self.device)
        self.inn = gc.GenerativeClassifier(init_latent_scale=mu_init, lr=lr_init).to(self.device)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.inn.optimizer, milestones=milestones, gamma=0.1)

        print("Device used for further computation is:", self.device)

        self.plot_columns = ['time', 'epoch', 'iteration',
                        'nll_joint_tr', 'nll_class_tr', 'cat_ce_tr',
                        'nll_joint_test', 'nll_class_test', 'cat_ce_test',
                        'acc_test', 'delta_mu_test']

        self.train_loss_names = [l for l in self.plot_columns if l[-3:] == '_tr']
        self.test_loss_names = [l for l in self.plot_columns if l[-4:] == '_test']

        self.header_fmt = '{:>15}' * len(self.plot_columns)
        self.output_fmt = '{:15.1f}      {:04d}/{:04d}' + '{:15.5f}' * (len(self.plot_columns) - 3)


    def get_dataset(self, pin_memory=True, drop_last=True):
        """
        Init train-, testset and train-, testloader for experiment. Furthermore criterion will be initialized.

        :param dataset: string that describe which dataset to use for training. Current Options: "mnist", "cifar"
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
        :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
        """
        print()
        print("Loading Dataset:")
        self.trainset, self.testset, self.classes = dl.load_imagenet()
        self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
        self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        self.num_classes = len(self.classes)


    def train(self):
        """
        Train INN model.
        """

        self.train_loss_log = {l: [] for l in self.train_loss_names}
        self.test_loss_log = {l: [] for l in self.test_loss_names}

        t_start = time()

        for epoch in range(self.num_epoch):
            self.scheduler.step()
            self.inn.train()

            running_avg = {l: [] for l in self.train_loss_names}

            print()
            print(80 * '-')
            print()

            print("Epoch: {}".format(epoch + 1))
            print("Training:")

            if self.device != torch.device("cuda"):
                print("Warning: GPU is not used for this computation")
                print("device:", self.device)

            for i, data in enumerate(tqdm(self.trainloader), 0):
                img, labels = data
                img, labels = img.to(self.device), self.onehot(labels.to(self.device))

                self.inn.optimizer.zero_grad()

                feat = self.vgg(img)
                
                if i == 1:
                    print(feat.size())

                losses = self.inn(feat.flatten(), labels)
                loss = losses['nll_joint_tr'] + self.beta * losses['cat_ce_tr']

                loss.backward()
                self.inn.optimizer.step()

                for name in self.train_loss_names:
                    running_avg[name].append(losses[name].item())

            if (epoch % self.interval_log) == 0:
                for name in self.train_loss_names:
                    running_avg[name] = np.mean(running_avg[name])
                    self.train_loss_log[name].append(running_avg[name])

                test_x = torch.stack([x[0][0] for x in self.testset], dim=0).to(self.device)
                test_y = self.onehot(torch.LongTensor([x[1] for x in self.testset]).to(self.device))
                test_losses = self.inn.validate(test_x, test_y)

                for name in self.test_loss_names:
                    running_avg[name] = test_losses[name].item()
                    self.test_loss_log[name].append(running_avg[name])

                losses_display = [(time() - t_start) / 60., epoch, self.num_epoch]
                losses_display += [running_avg[l] for l in self.plot_columns[3:]]
                print(self.output_fmt.format(*losses_display), flush=True)

            if epoch > 0 and (epoch % self.interval_checkpoint) == 0:
                self.inn.save(f'{self.modelname}_{epoch}.pt')

            if (epoch % self.interval_figure) == 0:
                # TODO: evaluater
                pass

        print()
        print(80 * "#")
        print(80 * "#")
        print()

        print("Evaluating:")
        self.inn.eval()
        test_x = torch.stack([x[0][0] for x in self.testset], dim=0).to(self.device)
        test_y = self.onehot(torch.LongTensor([x[1] for x in self.testset]).to(self.device))
        test_losses = self.inn.validate(test_x, test_y)
        self.test_acc = test_losses['acc_test']

        print("Final Test Accuracy:", self.test_acc)

        self.inn.save(f'{self.modelname}.pt')
        fm.save_variable(self.test_acc, '{}_acc'.format(self.modelname))
        fm.save_variable([self.train_loss_log, self.test_loss_log], '{}_loss'.format(self.modelname))


    def onehot(self, label):
        y = torch.cuda.FloatTensor(label.shape[0], self.num_classes).zero_()
        y.scatter_(1, label.view(-1, 1).long(), 1.)
        return y


    def load_inn(self, epoch=None):
        """
        Load pre-trained model based on modelname.

        :return: None
        """
        if epoch is None:
            self.inn.load(f'{self.modelname}.pt')
        else:
            self.inn.load(f'{self.modelname}_{epoch}.pt')


    def load_variables(self):
        """
        Load recorded loss and accuracy training history to class variable.

        :return: None
        """
        self.test_acc = fm.load_variable('{}_acc'.format(self.modelname))
        self.train_loss_log, self.test_loss_log = fm.load_variable('{}_loss'.format(self.modelname))


    def print_accuracy(self):
        """
        Plot train and test accuracy during training.

        :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
        :param figsize: the size of the generated plot
        :param font_size: font size of labels
        :param y_log_scale: y axis will have log scale instead of linear
        :return: None
        """

        self.load_variables()

        print("Final Test Accuracy:", self.test_acc)


    def plot_loss(self, sub_dim=None, figsize=(15, 10), font_size=24, y_log_scale=False):
        """
        Plot train and test loss during training.

        :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
        :param figsize: the size of the generated plot
        :param font_size: font size of labels
        :param y_log_scale: y axis will have log scale instead of linear
        :return: None
        """

        self.load_variables()

        pl.plot([x for x in range(1, self.num_epoch+1, self.interval_log)],
                [self.train_loss_log[name] for name in self.train_loss_names], 'Epoch', 'Loss',
                ['{}'.format(name) for name in self.train_loss_names], "Train Loss History {}".format(self.modelname),
                "train_loss_{}".format(self.modelname), sub_dim, figsize, font_size, y_log_scale)

        for name in self.train_loss_names:
            pl.plot([x for x in range(1, self.num_epoch+1, self.interval_log)], self.train_loss_log[name], 'Epoch',
                    'Loss', ['loss'], "{} Train Loss History {}".format(name, self.modelname),
                    "train_loss_{}_{}".format(self.modelname, name), sub_dim, figsize, font_size, y_log_scale)

        pl.plot([x for x in range(1, self.num_epoch+1, self.interval_log)],
                [self.test_loss_log[name] for name in self.test_loss_names], 'Epoch', 'Loss',
                ['{}'.format(name) for name in self.test_loss_names], "Test Loss History {}".format(self.modelname),
                "test_loss_{}".format_map(self.modelname), sub_dim, figsize, font_size, y_log_scale)

        for name in self.test_loss_names:
            pl.plot([x for x in range(1, self.num_epoch+1, self.interval_log)], self.test_loss_log[name], 'Epoch',
                    'Loss', ['loss'], "{} Test Loss History {}".format(name, self.modelname),
                    "test_loss_{}_{}".format(self.modelname, name), sub_dim, figsize, font_size, y_log_scale)







