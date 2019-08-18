from FrEIA import framework as fr
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
import torchvision.models as models


def get_vgg16():
    """
    Get features from pretrained VGG16 model.

    :return: partial VGG16 model, which takes ImageNet images and return features
    """
    vgg = models.vgg16(pretrained=True)
    vgg_feature = vgg.features

    return vgg_feature



def inn_model(img_dims=[25088]):
    """
    Return INN model.

    :param img_dims: size of the model input images. Default: Size of MNIST images
    :return: INN model
    """

    inp = fr.InputNode(*img_dims, name='input')

    fc1 = fr.Node([inp.out0], la.GLOWCouplingBlock,
                  {'F_class': fu.F_fully_connected, 'F_args': {'internal_size': 2048}, 'clamp': 2}, name='fc1')

    fc2 = fr.Node([fc1.out0], la.GLOWCouplingBlock,
                  {'F_class': fu.F_fully_connected, 'F_args': {'internal_size': 2048}, 'clamp': 2}, name='fc2')

    fc3 = fr.Node([fc2.out0], la.GLOWCouplingBlock,
                  {'F_class': fu.F_fully_connected, 'F_args': {'internal_size': 2048}, 'clamp': 2}, name='fc3')

    outp = fr.OutputNode([fc3.out0], name='output')

    nodes = [inp, outp, fc1, fc2, fc3]

    model = fr.ReversibleGraphNet(nodes, 0, 1)

    return model


