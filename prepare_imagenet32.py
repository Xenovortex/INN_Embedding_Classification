import os
import pickle
import urllib.request as req
from PIL import Image
import numpy as np

class_dict_link = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
classes = pickle.load(req.urlopen(class_dict_link))

#mean_image = None
counter = 0

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(datafolder, idx, img_size=32):
    """
    Load data batch from downloaded imagenet64. If mean is provided, load testset and apply mean for normalization.
    """
    global counter

    if idx is not None:
        datafile = os.path.join(datafolder, 'train_data_batch_')
        d = unpickle(datafile + str(idx))
        #mean_image = d['mean']
    else:
        datafile = os.path.join(datafolder, 'val_data')
        d = unpickle(datafile)

    x = d['data']
    y = d['labels']

    #x = x / np.float32(255)
    #mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i - 1 for i in y])

    #x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    #x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)


    if idx is not None:
        for i, img in enumerate(x):

            pil_img = Image.fromarray(img.astype(np.uint8))
            label = y[i]
            class_label = classes[label]

            if not os.path.exists(os.path.join('./datasets/ImgFolder_imagenet32/train', class_label)):
                os.mkdir(os.path.join('./datasets/ImgFolder_imagenet32/train', class_label))

            image_name =  str(counter).zfill(7) + ".png"
            path_to = os.path.join('./datasets/ImgFolder_imagenet32/train', class_label, image_name)
            pil_img.save(path_to)
            print("Image: {} saved in {}".format(image_name, path_to))
            counter += 1
    else:
        for i, img in enumerate(x):
            pil_img = Image.fromarray(img)
            label = y[i]
            class_label = classes[label]

            if not os.path.exists(os.path.join('./datasets/ImgFolder_imagenet32/test', class_label)):
                os.mkdir(os.path.join('./datasets/ImgFolder_imagenet32/test', class_label))

            image_name = str(counter).zfill(7) + ".png"
            path_to = os.path.join('./datasets/ImgFolder_imagenet32/test', class_label, image_name)
            pil_img.save(path_to)
            print("Image: {} saved in {}".format(image_name, path_to))
            counter += 1


save_path = "./datasets/imagenet32"

if not os.path.exists('./datasets/ImgFolder_imagenet32'):
    os.mkdir('./datasets/ImgFolder_imagenet32')

if not os.path.exists('./datasets/ImgFolder_imagenet32/train'):
    os.mkdir('./datasets/ImgFolder_imagenet32/train')

if not os.path.exists('./datasets/ImgFolder_imagenet32/test'):
    os.mkdir('./datasets/ImgFolder_imagenet32/test')


for i in range(1, 11):
    load_databatch(save_path, idx=i)

load_databatch(save_path, idx=None)

print("{} Images saved.".format(counter))