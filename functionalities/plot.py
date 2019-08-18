import os
import numpy as np
import matplotlib.pyplot as plt
import torch



def plot(x, y, x_label, y_label, plot_label, title, filename, sub_dim=None, figsize=(15, 10), font_size=24,
         y_log_scale=False):
    """
    Generate a plot based on given arguments. If y is a 2d numpy array/list, multiple plots will be generated within one
    diagram. If additionally x is also a 2d numpy array/list, multiple subplots will be generated.

    :param x: numpy array/list of x values to plot. If multiple subplots should be generated then x will be a 2d numpy
    array/list.
    :param y: numpy array/list of corresponding y values to plot. If multiple plots should be generated then y will be a
    2d numpy array/list.
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    :param plot_label: label for plots (appears in legend)
    :param title: title for the plot. Should be a list if multiple plots are generated.
    :param filename: file name under which the plot will be saved.
    :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
    :param figsize: the size of the generated plot
    :param font_size: font size of labels
    :param y_log_scale: y axis will have log scale instead of linear
    :return: None
    """

    plt.rcParams.update({'font.size': font_size})

    if not ('numpy' in str(type(x))):
        try:
            x = np.array(x)
        except TypeError:
            print("x is neither a numpy array nor a python list")

    if not ('numpy' in str(type(y))):
        try:
            y = np.array(y)
        except TypeError:
            print("y is neither a numpy array nor a python list")

    dim_x = len(x.shape)
    dim_y = len(y.shape)

    if (dim_x != 1 and dim_x != 2) or (dim_y != 1 and dim_y != 2) or (dim_x == 2 and dim_y == 1):
        raise ValueError("x has dimension {} and y has dimension {}".format(dim_x, dim_y))

    if dim_x == 1 and dim_y == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(x, y, label=plot_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_log_scale == True:
            ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True)
    elif dim_x == 1 and dim_y == 2:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for i, y_part in enumerate(y):
            ax.plot(x, y_part, label=plot_label[i])

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_log_scale == True:
            ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    elif dim_x == 2 and dim_y == 2:
        if sub_dim[0] * sub_dim[1] != len(y) or sub_dim[0] * sub_dim[1] != len(x):
            raise ValueError(
                "sub_dim dimension {} does not match dimension of x {} or y {}".format(sub_dim, len(y), len(x)))
        fig, ax = plt.subplots(sub_dim[0], sub_dim[1], figsize=figsize)

        counter = 0
        for i in range(sub_dim[0]):
            for j in range(sub_dim[1]):
                ax[i, j].plot(x[counter], y[counter], label=plot_label[counter])
                ax[i, j].set_xlabel(x_label[counter])
                ax[i, j].set_ylabel(y_label[counter])
                if y_log_scale == True:
                    ax.set_yscale('log')
                ax[i, j].set_title(title[counter])
                ax[i, j].grid(True)
                counter += 1

    plt.tight_layout()

    subdir = "./plot"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    fig.savefig(os.path.join(subdir, filename + ".png"), transparent=True, bbox_inches='tight', pad_inches=0)

    plt.show()


def imshow(img, figsize=(30, 30), filename=None):
    """
    Custom modified imshow function.
    :param img: image to plot
    :param figsize: the size of the generated plot
    :param filename: file name under which the plot will be saved. (optional)
    :return: None
    """
    img = torch.clamp(img, 0, 1)
    img = img.to('cpu')
    npimg = img.numpy()
    plt.figsize = figsize
    if len(img.shape) == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    elif len(img.shape) == 2:
        plt.imshow(npimg, cmap='gray')
    else:
        print("Plotting image with dimension {} is not implemented/possible.".format(img.shape))

    if filename is not None:
        subdir = "./plot"
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        plt.savefig(os.path.join(subdir, filename + ".png"),  transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.show()
