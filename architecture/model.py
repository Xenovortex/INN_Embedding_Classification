from FrEIA import framework as fr
from FrEIA.modules import coupling_layers as la
import torchvision.models as models
import torch.nn as nn


def get_vgg16():
    """
    Get features from pretrained VGG16 model.

    :return: partial VGG16 model, which takes ImageNet images and return features
    """
    vgg = models.vgg16(pretrained=True)
    vgg_feature = vgg.features[:-1]

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







