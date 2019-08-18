import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import urllib2


def load_imagenet():
    """
    Check if ImageNet dataset already exists in directory "/datasets/imagenet". If not, the ImageNet dataset is
    downloaded.

    :return: trainset, testset and classes of ImageNet
    """

    save_path = "./datasets/imagenet"
    class_dict_link = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.ImageNet(root=save_path, split='train', download=True, transform=transform)
    testset = datasets.ImageNet(root=save_path, split='val', download=True, transform=transform)

    classes = pickle.load(urllib2.urlopen(class_dict_link))

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