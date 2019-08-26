import os
from time import time
import glob
import torch
import numpy as np
from functionalities import dataloader as dl
from functionalities import filemanager as fm
from functionalities import plot as pl
from architecture import generative_classifier as gc
from architecture import model as m
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA



class inn_experiment:
    """
    Class for training INN models
    """

    def __init__(self, num_epoch, batch_size, milestones, modelname, device='cpu', lr_init=5e-4, mu_init=11., beta=5.0,
                 interval_log=1, interval_checkpoint=1, interval_figure=1, use_vgg=False):
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
                if not os.path.exists('./models'):
                    os.mkdir('./models')
                self.inn.save(f'models/{self.modelname}_{epoch}.pt')

            if (epoch % self.interval_figure) == 0:
                if not os.path.exists('./plots'):
                    os.mkdir('./plots')
                self.val_plots0(f'plots/{self.modelname}_{epoch}.pdf')


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


    def show_samples(self, y, T=0.75):
        with torch.no_grad():
            samples = self.inn.sample(y, T).cpu().numpy()

        w = y.shape[1]
        h = int(np.ceil(y.shape[0] / w))

        plt.figure()
        for k in range(y.shape[0]):
            plt.subplot(h, w, k + 1)
            plt.imshow(samples[k], cmap='gray')
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()


    def show_latent_space(self):
        ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

        clusters = self.inn.mu.data.cpu().numpy().squeeze()
        pca = PCA(n_components=2)
        pca.fit(clusters)

        mu_red = pca.transform(clusters)
        z_red = []
        true_label = []

        with torch.no_grad():
            for x, y in self.testloader:
                true_label.append(y.cpu().numpy())
                x, y = x.cuda(), self.onehot(y.cuda())
                z = self.inn(x).cpu().numpy()
                z_red.append(pca.transform(z))

        z_red = np.concatenate(z_red, axis=0)
        true_label = np.concatenate(true_label, axis=0)

        plt.figure()
        plt.scatter(mu_red[:, 0], mu_red[:, 1], c=np.arange(10), cmap='tab10', s=250, alpha=0.5)
        plt.scatter(z_red[:, 0], z_red[:, 1], c=true_label, cmap='tab10', s=1)
        plt.tight_layout()

    def outlier_histogram(self):
        ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

        score, correct_pred = [], []

        from torchvision.datasets import FashionMNIST
        from torch.utils.data import DataLoader
        fashion_generator = DataLoader(
            FashionMNIST('./fashion_mnist', download=True, train=False, transform=data.transform),
            batch_size=self.batch_size, num_workers=8)

        # continue using dropout for WAIC
        self.inn.train()

        def waic(x):
            waic_samples = 1
            ll_joint = []
            for i in range(waic_samples):
                if self.vgg is not None:
                    feat = self.vgg(x)
                else:
                    feat = x
                losses = self.inn(feat, y=None, loss_mean=False)
                ll_joint.append(losses['nll_joint_tr'].cpu().numpy())

            ll_joint = np.stack(ll_joint, axis=1)
            return np.mean(ll_joint, axis=1) + np.var(ll_joint, axis=1)

        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.cuda(), self.onehot(y.cuda())
                score.append(waic(x))
                if self.vgg is not None:
                    feat = self.vgg(x)
                else:
                    feat = x
                losses = self.inn(feat, y, loss_mean=False)
                correct_pred.append((torch.argmax(y, dim=1)
                                     == torch.argmax(losses['logits_tr'], dim=1)).cpu().numpy())

            score_fashion = []
            for x, y in fashion_generator:
                x = x.cuda()
                score_fashion.append(waic(x))

            #score_adv = []
            #score_adv_ref = []

            #adv_images = np.stack([np.load(f) for f in glob.glob('./adv_examples/adv_*.npy')], axis=0)
            #ref_images = np.stack([np.load(f) for f in glob.glob('./adv_examples/img_*.npy')], axis=0)

            #adv_images = torch.Tensor(adv_images).cuda()
            #ref_images = torch.Tensor(ref_images).cuda()

            #score_adv = waic(adv_images)
            #score_ref = waic(ref_images)

        self.inn.eval()

        score = np.concatenate(score, axis=0)
        correct_pred = np.concatenate(correct_pred, axis=0)
        score_fashion = np.concatenate(score_fashion, axis=0)

        # val_range = [np.quantile(np.concatenate((score, score_fashion, score_adv)),  0.01),
        # np.quantile(np.concatenate((score, score_fashion, score_adv)),  0.6)]
        val_range = [-8, 0]
        val_range[0] -= 0.03 * (val_range[1] - val_range[0])
        val_range[1] += 0.03 * (val_range[1] - val_range[0])

        bins = 40

        plt.figure()
        plt.hist(score[correct_pred == 1], bins=bins, range=val_range, histtype='step', label='correct', density=True,
                 color='green')
        plt.hist(score[correct_pred == 0], bins=3 * bins, range=val_range, histtype='step', label='not correct',
                 density=True, color='red')
        plt.hist(score_fashion, bins=bins, range=val_range, histtype='step', label='$\mathcal{Fashion}$', density=True,
                 color='magenta')

        #plt.hist(score_adv, bins=bins, range=val_range, histtype='step', label='Adv attacks', density=True,
         #        color='blue')
        #plt.hist(score_ref, bins=bins, range=val_range, histtype='step', label='Non-attacked images', density=True,
         #        color='gray')

        plt.legend()


    def calibration_curve(self):

        pred = []
        gt = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.cuda(), self.onehot(y.cuda())
                if self.vgg is not None:
                    feat = self.vgg(x)
                else:
                    feat = x
                logits = self.inn(feat, y, loss_mean=False)['logits_tr']

                # log_pred = logits - torch.logsumexp(logits + np.log(1/10.), dim=1, keepdim=True)
                # exp_pred = torch.exp(log_pred)
                # print(torch.mean(torch.max(exp_pred, dim=1)[0]))

                pred.append(torch.softmax(logits, dim=1).cpu().numpy())
                # pred.append(torch.exp(log_pred).cpu().numpy())
                gt.append(y.cpu().numpy())

        pred = np.concatenate(pred, axis=0).flatten()
        gt = np.concatenate(gt, axis=0).astype(np.bool).flatten()

        mask = (pred > 1e-6)
        mask = mask * (pred < (1 - 1e-6))

        pred, gt = pred[mask], gt[mask]

        n_bins = np.sum(mask) / 50.
        pred_bins = np.quantile(pred, np.linspace(0., 1., n_bins))
        print(np.sum(mask))

        correct = pred[gt]
        wrong = pred[np.logical_not(gt)]

        hist_correct, _ = np.histogram(correct, bins=pred_bins)
        hist_wrong, _ = np.histogram(wrong, bins=pred_bins)

        q = hist_correct / (hist_wrong + hist_correct)
        p = 0.5 * (pred_bins[1:] + pred_bins[:-1])

        poisson_err = q * np.sqrt(1 / hist_correct + 1 / (hist_wrong + hist_correct))

        plt.figure(figsize=(10, 10))
        plt.errorbar(p, q, yerr=poisson_err, capsize=4, fmt='-o')
        plt.fill_between(p, q - poisson_err, q + poisson_err, alpha=0.25)
        plt.plot([0, 1], [0, 1], color='black')

    def val_plots(self, fname):
        n_classes = self.num_classes
        n_samples = 4

        y_digits = torch.zeros(n_classes * n_samples, n_classes).cuda()
        for i in range(n_classes):
            y_digits[n_samples * i: n_samples * (i + 1), i] = 1.

        self.show_samples(y_digits)
        self.show_latent_space()

        with PdfPages(fname) as pp:
            figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                fig.savefig(pp, format='pdf')

        plt.close('all')









