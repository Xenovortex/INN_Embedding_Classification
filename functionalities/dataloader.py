import os
import torch
import torch.utils.data
import pickle
import urllib.request as req
import numpy as np
from tqdm import tqdm_notebook as tqdm


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(datafolder, idx, img_size=64, mean=None):
    """
    Load data batch from downloaded imagenet64. If mean is provided, load testset and apply mean for normalization.
    """
    if mean is None:
        datafile = os.path.join(datafolder, 'train_data_batch_')
        d = unpickle(datafile + str(idx))
        mean_image = d['mean']
    else:
        datafile = os.path.join(datafolder, 'val_data')
        d = unpickle(datafile)
        mean_image = mean

    x = d['data']
    y = d['labels']
    
    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    X = x[0:data_size, :, :, :]
    Y = y[0:data_size]
    
    if mean is None:
        return torch.Tensor(X), torch.Tensor(Y), mean_image
    else:
        return torch.Tensor(X), torch.Tensor(Y)


def load_imagenet():
    """
    Check if ImageNet dataset already exists in directory "/datasets/imagenet". If not, the ImageNet dataset is
    downloaded.

    :return: trainset, testset and classes of ImageNet
    """

    save_path = "./datasets/imagenet"
    class_dict_link = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'

    train_x = []
    train_y = []
    mean_image = None
    for i in tqdm(range(1, 11)):
        x, y, mean_image = load_databatch(save_path, idx=i)
        train_x.append(x)
        train_y.append(y)

    train_x, train_y = torch.cat(train_x, out=torch.Tensor(len(train_x), train_x[0].size()[0], train_x[0].size()[1], train_x[0].size()[2])), torch.cat(train_y, out=torch.Tensor(len(train_y)))
    trainset = torch.utils.data.TensorDataset(train_x, train_y)

    test_x, test_y = load_databatch(save_path, idx=None, mean=mean_image)
    testset = torch.utils.data.TensorDataset(test_x, test_y)

    classes = pickle.load(req.urlopen(class_dict_link))

    return trainset, testset, classes


def get_loader(dataset, batch_size, pin_memory=True, drop_last=True):
    """
    Create loader for a given dataset.

    :param dataset: dataset for which a loader will be created
    :param batch_size: size of the batch the loader will load during training
    :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
    :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
    :return: loader
    """

    loader = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size, drop_last=drop_last)

    return loader


def split_dataset(dataset, ratio, batch_size, pin_memory=True, drop_last=True):
    """
    Split a dataset into two subset. e.g. trainset and validation-/testset
    :param dataset: dataset, which should be split
    :param ratio: the ratio the two splitted datasets should have to each other
    :param batch_size: batch size the returned dataloaders should have
    :param pin_memory: pin_memory argument for pytorch dataloader, will be simply forwarded
    :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
    :return: dataloader_1, dataloader_2
    """

    indices = torch.randperm(len(dataset))
    idx_1 = indices[:len(indices) - int(ratio * len(indices))]
    idx_2 = indices[len(indices) - int(ratio * len(indices)):]

    dataloader_1 = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_1),
                                               drop_last=drop_last)

    dataloader_2 = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_2),
                                               drop_last=drop_last)

    return dataloader_1, dataloader_2
