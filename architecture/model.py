from FrEIA import framework as fr
from FrEIA import modules as la
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np


def get_vgg16():
    """
    Get features from pretrained VGG16 model.

    :return: partial VGG16 model, which takes ImageNet images and return feature0
    """
    vgg = models.vgg16(pretrained=True)
    vgg_feature = vgg.features[:17]

    return vgg_feature

def fc_constr(c_in, c_out):
    net = [nn.Linear(c_in, 2048), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(2048, c_out), ]
    net[-1].weight.data *= 0.1
    return nn.Sequential(*net)


def inn_model(img_dims=4):
    """
    Return INN model.

    :param img_dims: size of the model input images. Default: Size of MNIST images
    :return: INN model
    """

    inp = fr.InputNode(img_dims, name='input')

    fc1 = fr.Node([inp.out0], la.GLOWCouplingBlock, {'subnet_constructor': fc_constr, 'clamp': 2}, name='fc1')
    fc2 = fr.Node([fc1.out0], la.GLOWCouplingBlock, {'subnet_constructor': fc_constr, 'clamp': 2}, name='fc2')
    fc3 = fr.Node([fc2.out0], la.GLOWCouplingBlock, {'subnet_constructor': fc_constr, 'clamp': 2}, name='fc3')

    outp = fr.OutputNode([fc3.out0], name='output')

    nodes = [inp, outp, fc1, fc2, fc3]

    model = fr.ReversibleGraphNet(nodes)

    return model


fc_width = 512
conv_width = 256
n_coupling_blocks_fc = 12
n_coupling_blocks_conv_0 = 2
n_coupling_blocks_conv_1 = 2
clamp = 2.0


def conv_constr_0(c_in, c_out):
    net = [nn.Conv2d(c_in, conv_width//2, 3, padding=1),
           nn.BatchNorm2d(conv_width//2, track_running_stats=False),
           nn.ReLU(),
           nn.Conv2d(conv_width//2,  c_out, 3, padding=1)]

    net[-1].weight.data *= 0.003
    return nn.Sequential(*net)

def conv_constr_1(c_in, c_out):
    net = [nn.Conv2d(c_in, conv_width, 3, padding=1),
           nn.BatchNorm2d(conv_width, track_running_stats=False),
           nn.Dropout2d(p=0.2),
           nn.ReLU(),
           nn.Conv2d(conv_width,  c_out, 3, padding=1)]

    net[-1].weight.data *= 0.003
    return nn.Sequential(*net)

def fc_constr(c_in, c_out):
    net = [nn.Linear(c_in, fc_width),
           nn.ReLU(),
           nn.Dropout(p=0.2),
           nn.Linear(fc_width,  c_out), ]
    net[-1].weight.data *= 0.1
    return nn.Sequential(*net)

def random_orthog(n):
    w = np.zeros((n,n))
    for i,j in enumerate(np.random.permutation(n)):
        w[i,j] = 0.25
    return torch.FloatTensor(w)


def constuct_inn(img_dims=[3, 32, 32]):

    split_nodes = []

    nodes = [fr.InputNode(*img_dims, name='input')]

    nodes.append(fr.Node(nodes[-1].out0, la.Reshape, {'target_dim':(img_dims[0], *img_dims[1:])}, name='reshape'))
    nodes.append(fr.Node(nodes[-1].out0, la.HaarDownsampling, {'rebalance':0.5}, name='haar_down_1'))

    for k in range(n_coupling_blocks_conv_0):
        nodes.append(fr.Node(nodes[-1], la.GLOWCouplingBlock, {'subnet_constructor':conv_constr_0, 'clamp':clamp}, name=f'CB CONV_0_{k}'))
        if k%2: nodes.append(fr.Node(nodes[-1], la.Fixed1x1Conv, {'M':random_orthog(12)}))

    nodes.append(fr.Node(nodes[-1].out0, la.HaarDownsampling, {'rebalance':0.5}, name='haar_down_2'))

    for k in range(n_coupling_blocks_conv_1):
        nodes.append(fr.Node(nodes[-1], la.GLOWCouplingBlock, {'subnet_constructor':conv_constr_1, 'clamp':clamp}, name=f'CB CONV_1_{k}'))
        if k%2: nodes.append(fr.Node(nodes[-1], la.Fixed1x1Conv, {'M':random_orthog(48)}))

    nodes.append(fr.Node(nodes[-1].out0, la.Flatten, {}, name='flatten'))

    for k in range(n_coupling_blocks_fc):
        nodes.append(fr.Node(nodes[-1], la.GLOWCouplingBlock, {'subnet_constructor':fc_constr, 'clamp':clamp}, name=f'CB FC_{k}'))
        if k%2: nodes.append(fr.Node(nodes[-1], la.PermuteRandom, {'seed':k}))

    nodes.append(fr.OutputNode(nodes[-1], name='output'))

    return fr.ReversibleGraphNet(nodes)







