import os
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from deepml.utils import get_random_samples_batch_from_loader, transform_input, transform_target


def plot_images(images, labels=None, cols=4, figsize=(10, 10), fontsize=14):

    plt.figure(figsize=figsize)
    rows = int(np.ceil(len(images) / cols))
    for index, image in enumerate(images):
        ax = plt.subplot(rows, cols, index + 1,  xticks=[], yticks=[])
        if labels:
            ax.set_title(labels[index])
        ax.title.set_fontsize(fontsize)
        plt.imshow(image)
    plt.tight_layout()


def plot_images_with_title(image_title_generator, samples, cols=4, figsize=(10, 10), fontsize=14):
    """
    Plots images with colored title.
    Accepts generator that yields triplet tuple (image, title, title color)

    :param image_title_generator:
    :param samples:
    :param cols:
    :param figsize:
    :param fontsize:
    :return:
    """

    plt.figure(figsize=figsize)
    rows = int(np.ceil(samples / cols))
    for index, (image, title, title_color) in enumerate(image_title_generator):
        ax = plt.subplot(rows, cols, index + 1,  xticks=[], yticks=[])
        ax.set_title(title, color=mpl.rcParams['text.color'] if title_color is None else title_color)
        ax.title.set_fontsize(fontsize)
        plt.imshow(image)
    plt.tight_layout()


def show_images_from_loader(loader, image_inverse_transform=None, samples=9, cols=3, figsize=(5, 5),
                            classes=None, title_color=None):
    x, y = get_random_samples_batch_from_loader(loader, samples=samples)
    x = transform_input(x, image_inverse_transform)

    image_title_generator = ((x[index], transform_target(y[index], classes),
                              title_color) for index in range(x.shape[0]))
    plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)


def show_images_from_folder(img_dir, samples=9, cols=3, figsize=(10, 10), title_color=None):
    files = os.listdir(img_dir)
    if samples < len(files):
        samples = np.random.choice(files, size=samples, replace=False)
    else:
        samples = files

    image_generator = ((Image.open(os.path.join(img_dir, file)), file, title_color)
                       for file in samples)
    plot_images_with_title(image_generator, len(samples), cols=cols, figsize=figsize)