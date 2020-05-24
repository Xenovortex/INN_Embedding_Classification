import numpy as np
import torch
from functionalities import inn_loss as il
from functionalities import dataloader as dl
from tqdm import tqdm_notebook as tqdm


class inn_experiment:
    """
    Class for training INN models
    """


    def __init__(self, num_epoch, batch_size, lr_init, milestones, get_model, modelname, num_classes, device='cpu',
                 weight_decay=1e-6):
        """
        Init class with pretraining setup.

        :param num_epoch: number of training epochs
        :param batch_size: Batch Size to use during training.
        :param lr_init: Starting learning rate. Will be decrease with adaptive learning.
        :param milestones: list of training epochs at which the learning rate will be reduce
        :param get_model: function that returns a model for training
        :param modelname: model name under which the model will be saved
        :param device: device on which to do the computation (CPU or CUDA). Please use get_device() to get device
        variable, if using multiple GPU's.
        :param weight_decay: weight decay (L2 penalty) for adam optimizer
               """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.modelname = modelname
        self.num_classes = num_classes
        self.device = device

        self.model = get_model().to(self.device)
        self.init_param()

        self.model_params = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                self.model_params.append(parameter)

        self.optimizer = torch.optim.Adam(self.model_params, lr=lr_init, betas=(0.8, 0.8), eps=1e-04,
                                          weight_decay=weight_decay)
        self.criterion = il.INN_loss()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)


    def get_dataset(self, dataset, pin_memory=True, drop_last=True):
        """
        Init train-, testset and train-, testloader for experiment.

        :param dataset: string that describe which dataset to use for training. Current Options: "mnist", "cifar"
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
        :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
        """
        if dataset == "mnist":
            self.trainset, self.testset, self.classes = dl.load_mnist()
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        elif dataset == "cifar":
            self.trainset, self.testset, self.classes = dl.load_cifar()
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        else:
            print("The requested dataset is not implemented yet.")


    def train(self):
        """
        Train INN model.
        """

        self.tot_train_log = []
        self.tot_test_log = []
        self.lx_train_log = []
        self.lx_test_log = []
        self.ly_train_log = []
        self.ly_test_log = []
        self.lz_train_log = []
        self.lz_test_log = []

        for epoch in range(self.num_epoch):
            self.scheduler.step()
            self.model.train()

            losses = np.zeros(5, dtype=np.double)

            print("Epoch: {}",format(epoch + 1))
            print("Training:")

            for i, data in enumerate(tqdm(self.trainloader), 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                lat_img = self.model(inputs)
                lat_shape = lat_img.shape
                lat_img = lat_img.view(lat_img.size(0), -1)
                binary_label = lat_img.new_zeros(lat_img.size(0), self.num_classes)
                idx = torch.arange(labels.size(0), dtype=torch.long)
                binary_label[idx, labels] = 1
                lat_img_mod = torch.cat([binary_label, lat_img[:, self.num_classes:]])
                lat_img_mod = lat_img_mod.view(lat_shape)
                output = self.model(lat_img_mod, rev=True)
                batch_loss = self.criterion(inputs, lat_img, output, labels)
                batch_loss[0].backward()
                self.optimizer.step()

                for i in range(len(batch_loss)):
                    losses[i] += batch_loss[i].item()




        for i, data in enumerate(tqdm(trainloader), 0):
            criterion.update_num_step(num_step)
            num_step += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            lat_img = model(inputs)
            lat_shape = lat_img.shape
            lat_img = lat_img.view(lat_img.size(0), -1)
            else:
                lat_img_mod = torch.cat([lat_img[:, :latent_dim], lat_img.new_zeros((lat_img[:, latent_dim:]).shape)],
                                        dim=1)

            lat_img_mod = lat_img_mod.view(lat_shape)

            output = model(lat_img_mod, rev=True)

            if conditional:
                batch_loss = criterion(inputs, lat_img, output, labels, binary_label)
            elif use_label:
                batch_loss = criterion(inputs, lat_img, output, labels)
            else:
                batch_loss = criterion(inputs, lat_img, output)

            batch_loss[0].backward()

            optimizer.step()

            for i in range(len(batch_loss)):
                losses[i] += batch_loss[i].item()

        correct = correct * 100. / len(trainloader.dataset)
        losses /= len(trainloader)
        tot_loss_log.append(losses[0])
        rec_loss_log.append(losses[1])
        dist_loss_log.append(losses[2])
        spar_loss_log.append(losses[3])
        disen_loss_log.append(losses[4])
        if len(losses) == 6:
            disc_loss_log.append(losses[5])
            print(
                'Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f} \t L_disc: {:.3f}'.format(
                    losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))
            print('Train Accuracy: {:.1f}'.format(correct))
        else:
            print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f}'.format(
                losses[0], losses[1], losses[2], losses[3], losses[4]))

        if validloader is not None:
            print('\n')
            print('Compute and record loss on validation set')
            valid_loss = ev.get_loss(validloader, model, criterion, latent_dim, tracker, conditional, disc_lst,
                                     use_label, device)
            tot_valid_loss_log.append(valid_loss[0])
            rec_valid_loss_log.append(valid_loss[1])
            dist_valid_loss_log.append(valid_loss[2])
            spar_valid_loss_log.append(valid_loss[3])
            disen_valid_loss_log.append(valid_loss[4])
            if len(valid_loss) == 6:
                disc_valid_loss_log.append(valid_loss[5])
                print(
                    'Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f} \t L_disc: {:.3f}'.format(
                        valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5]))
            else:
                print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f}'.format(
                    valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4]))

            print('latent image mean: {:.3f} \t latent image std: {:.3f}'.format(tracker.mu, tracker.std))

            if valid_loss[0] <= min_loss:
                last_best_epoch = best_epoch
                best_epoch = epoch + 1
                min_loss = valid_loss[0]
                fm.save_model(model, "{}_{}_best".format(modelname, best_epoch))
                fm.save_weight(model, "{}_{}_best".format(modelname, best_epoch))
                if last_best_epoch != 0:
                    fm.delete_file("models", "{}_{}_best".format(modelname, last_best_epoch))
                    fm.delete_file("weights", "{}_{}_best".format(modelname, last_best_epoch))

        if testloader is not None:
            print('\n')
            print('Compute and record loss on test set:')
            test_loss = ev.get_loss(testloader, model, criterion, latent_dim, tracker, conditional, disc_lst, use_label,
                                    device)
            tot_test_loss_log.append(test_loss[0])
            rec_test_loss_log.append(test_loss[1])
            dist_test_loss_log.append(test_loss[2])
            spar_test_loss_log.append(test_loss[3])
            disen_test_loss_log.append(test_loss[4])
            if len(test_loss) == 6:
                disc_test_loss_log.append(test_loss[5])
                print(
                    'Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f} \t L_disc: {:.3f}'.format(
                        test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5]))
            else:
                print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f}'.format(
                    test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]))

            print('latent image mean: {:.3f} \t latent image std: {:.3f}'.format(tracker.mu, tracker.std))

        if epoch % num_epoch_save == 0 or epoch == (num_epoch - 1):
            pl.plot_diff(model, testloader, latent_dim, device, num_img, grid_row_size,
                         filename=modelname + "_{}".format(epoch))

        print('\n')
        print('-' * 80)
        print('\n')

    if validloader is not None:
        print("Lowest Validation Loss: {:3f} was achieved at epoch: {}".format(min_loss, best_epoch))

    print("Finished Training")

    if subdir is None:
        subdir = modelname

    if save_model:
        model.to('cpu')
        fm.save_model(model, "{}_{}".format(modelname, num_epoch), subdir)
        fm.save_weight(model, "{}_{}".format(modelname, num_epoch), subdir)

    if save_variable:
        fm.save_variable([tot_loss_log, tot_valid_loss_log, tot_test_loss_log],
                         "total_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([rec_loss_log, rec_valid_loss_log, rec_test_loss_log],
                         "reconstruction_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([dist_loss_log, dist_valid_loss_log, dist_test_loss_log],
                         "distribution_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([spar_loss_log, spar_valid_loss_log, spar_test_loss_log],
                         "sparsity_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([disen_loss_log, disen_valid_loss_log, disen_test_loss_log],
                         "disentanglement_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([disc_loss_log, disc_valid_loss_log, disc_test_loss_log],
                         "discrete_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([tot_loss_log, rec_loss_log, dist_loss_log, spar_loss_log, disen_loss_log, disc_loss_log],
                         "train_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([tot_valid_loss_log, rec_valid_loss_log, dist_valid_loss_log, spar_valid_loss_log,
                          disen_valid_loss_log, disc_loss_log], "validation_loss_{}_{}".format(modelname, num_epoch),
                         subdir)
        fm.save_variable([tot_test_loss_log, rec_test_loss_log, dist_test_loss_log, spar_test_loss_log,
                          disen_test_loss_log, disc_loss_log], "test_loss_{}_{}".format(modelname, num_epoch), subdir)

    return model


    def init_param(self, sigma=0.1):
        """
        Initialize weights for INN models.

        :param sigma: standard deviation for gaussian
        :return:
        """
        for key, param in self.model.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = sigma * torch.randn(param.data.shape).cuda()
                if split[3][-1] == '3':  # last convolution in the coeff func
                    param.data.fill_(0.)