from FrEIA import framework as fr
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
from FrEIA.modules import reshapes as re


def mnist_inn_model(img_dims=[1, 28, 28]):
    """
    Return INN model for MNIST.

    :param img_dims: size of the model input images. Default: Size of MNIST images
    :return: INN model
    """

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                                                        'F_args': {'channels_hidden': 100}, 'clamp': 1}, name='conv1')

    conv2 = fr.Node([conv1.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                                                           'F_args': {'channels_hidden': 100}, 'clamp': 1},
                    name='conv2')

    conv3 = fr.Node([conv2.out0], la.glow_coupling_layer, {'F_class': fu.F_conv,
                                                           'F_args': {'channels_hidden': 100}, 'clamp': 1},
                    name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (img_dims[0] * img_dims[1] * img_dims[2],)}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer,
                 {'F_class': fu.F_small_connected, 'F_args': {'internal_size': 100}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (4, 14, 14)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, r1, r2, r3, r4, fc]

    model = fr.ReversibleGraphNet(nodes, 0, 1)

    return model

