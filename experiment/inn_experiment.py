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
                 interval_log=1, interval_checkpoint=1, interval_figure=20, use_vgg=False):
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

        if use_vgg:
            self.vgg = m.get_vgg16().to(self.device)
        else:
            self.vgg = None
        self.inn = gc.GenerativeClassifier(init_latent_scale=mu_init, lr=lr_init).to(self.device)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.inn.optimizer, milestones=milestones, gamma=0.1)

        print("Device used for further computation is:", self.device)

        self.plot_columns = ['time', 'epoch', 'iteration',
                        'nll_joint_tr', 'nll_class_tr', 'cat_ce_tr',
                        'nll_joint_test', 'nll_class_test', 'cat_ce_test',
                        'acc_test', 'delta_mu_test']

        self.train_loss_names = [l for l in self.plot_columns if l[-3:] == '_tr']
        self.test_loss_names = [l for l in self.plot_columns if l[-5:] == '_test']

        self.header_fmt = '{:>15}' * len(self.plot_columns)
        self.output_fmt = '{:15.1f}      {:04d}/{:04d}' + '{:15.5f}' * (len(self.plot_columns) - 3)


    def get_dataset(self, dataset='imagenet', pin_memory=True, num_workers=8):
        """
        Init train-, testset and train-, testloader for experiment. Furthermore criterion will be initialized.

        :param dataset: string that describe which dataset to use for training. Current Options: "mnist", "cifar"
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
        :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
        """
        print()
        print("Loading Dataset: {}".format(dataset))
        if dataset == "imagenet":
            self.trainset, self.testset, self.classes = dl.load_imagenet()
        elif dataset == "cifar":
            self.trainset, self.testset, self.classes = dl.load_cifar()
        else:
            print("The requested dataset is not implemented yet.")
            print("Possible options are: imagenet and cifar.")
        self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, shuffle=True, num_workers=num_workers)
        self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, shuffle=False, num_workers=num_workers)
        self.num_classes = len(self.classes)
        print("Finished!")


    def train(self):
        """
        Train INN model.
        """

        self.train_loss_log = {l: [] for l in self.train_loss_names}
        self.test_loss_log = {l: [] for l in self.test_loss_names}

        t_start = time()

        for epoch in range(self.num_epoch):
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

                if self.vgg is not None:
                    feat = self.vgg(img)
                else:
                    feat = img

                feat = feat.view(feat.size(0), -1)

                losses = self.inn(feat, labels)
                loss = losses['nll_joint_tr'] + self.beta * losses['cat_ce_tr']

                loss.backward()
                self.inn.optimizer.step()

                for name in self.train_loss_names:
                    running_avg[name].append(losses[name].item())

            self.scheduler.step()

            if (epoch % self.interval_log) == 0:
                self.inn.eval()
                print("Evaluating:")
                for name in self.train_loss_names:
                    running_avg[name] = np.mean(running_avg[name])
                    self.train_loss_log[name].append(running_avg[name])

                test_loss_lst = {l: [] for l in self.test_loss_names}
                for i, data in enumerate(tqdm(self.testloader), 0):
                    img, labels = data
                    img, labels = img.to(self.device), self.onehot(labels.to(self.device))

                    with torch.no_grad():
                        if self.vgg is not None:
                            feat = self.vgg(img)
                        else:
                            feat = img

                        feat = feat.view(feat.size(0), -1)

                        test_losses = self.inn.validate(feat, labels)

                        for name in self.test_loss_names:
                            test_loss_lst[name].append(test_losses[name].item())

                test_loss = {l: 0 for l in self.test_loss_names}
                for name in self.test_loss_names:
                    #test_loss[name] = sum(test_loss_lst[name]) / len(test_loss_lst[name])
                    test_loss[name] = np.mean(test_loss_lst[name])
                    running_avg[name] = test_loss[name].item()
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
        test_loss_lst = {l: [] for l in self.test_loss_names}
        for i, data in enumerate(tqdm(self.testloader), 0):
            img, labels = data
            img, labels = img.to(self.device), self.onehot(labels.to(self.device))

            with torch.no_grad():
                if self.vgg is not None:
                    feat = self.vgg(img)
                else:
                    feat = img

                feat = feat.view(feat.size(0), -1)

                test_losses = self.inn.validate(feat, labels)

                for name in self.test_loss_names:
                    test_loss_lst[name].append(test_losses[name].item())

        test_loss = {l: 0 for l in self.test_loss_names}
        for name in self.test_loss_names:
            test_loss[name] = np.mean(test_loss_lst[name])

        self.test_acc = test_loss['acc_test']

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







