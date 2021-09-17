import os
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from deepml.utils import get_random_samples_batch_from_loader, transform_input, transform_target, \
    get_random_samples_batch_from_dataset


def plot_images(images, labels=None, cols=4, figsize=(10, 10), fontsize=14):
    """
    Plot images with provided labels.
    :param images: The list of images of type np.array.
    :param labels: The corresponding labels for images.
    :param cols: The number of cols. Default is 4.
    :param figsize: The matplotlib figure size. Default is (10,10)
    :param fontsize: The fontsize for title display. Default is 14.
    :return: None
    """
    plt.figure(figsize=figsize)
    rows = int(np.ceil(len(images) / cols))
    for index, image in enumerate(images):
        ax = plt.subplot(rows, cols, index + 1, xticks=[], yticks=[])
        if labels:
            ax.set_title(labels[index])
        ax.title.set_fontsize(fontsize)
        plt.imshow(image)
    plt.tight_layout()


def plot_images_with_title(image_title_generator, samples, cols=4, figsize=(10, 10), fontsize=14):
    """
    Plots images with colored title.
    Accepts generator that yields triplet tuple (image: np.array, title: str, title color: str)

    :param image_title_generator: The generator returning tuples (image, title, title color)
    :param samples: The total number of samples in generator.
    :param cols: The number of columns. Default is 4.
    :param figsize: The matplotlib figure size. Default is (10,10)
    :param fontsize: The fontsize for title display. Default is 14.
    :return: None
    """

    plt.figure(figsize=figsize)
    rows = int(np.ceil(samples / cols))
    for index, (image, title, title_color) in enumerate(image_title_generator):
        ax = plt.subplot(rows, cols, index + 1, xticks=[], yticks=[])
        ax.set_title(title, color=mpl.rcParams['text.color'] if title_color is None else title_color)
        ax.title.set_fontsize(fontsize)
        plt.imshow(image)
    plt.tight_layout()


def show_images_from_loader(loader, image_inverse_transform=None, samples=9, cols=3, figsize=(5, 5),
                            classes=None, title_color=None):
    """
    Displays random samples of images from a torch.utils.data.DataLoader
   :param loader: An instance of torch.utils.data.DataLoader returning torch.tensor of shape in order #BCWH
   :param image_inverse_transform: The inverse transform to apply on image tensor before displaying it.
                                   Default is None.
                                   For imagenet normalized image tensor, use deepml.transforms.ImageNetInverseTransform
   :param samples: The number of random image samples to display. Default is 9.
   :param cols: The number of display columns in the matplotlib figure. Default is 3.
   :param figsize: The matplotlib figure size. Default is (10,10)
   :param classes: The list of class names for class indices return by torch dataset.
   :param title_color: The title color for images.
   :return: None
   """
    x, y = get_random_samples_batch_from_loader(loader, samples=samples)
    x = transform_input(x, image_inverse_transform)

    if not classes and hasattr(loader.dataset, 'classes'):
        classes = loader.dataset.classes

    image_title_generator = ((x[index], transform_target(y[index], classes),
                              title_color) for index in range(x.shape[0]))
    plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)


def show_images_from_dataset(dataset, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10),
                             classes=None, title_color=None):
    """
    Displays random samples of images from a torch.utils.data.Dataset
   :param dataset: An instance of torch.utils.data.Dataset returning torch.tensor of shape in order #BCWH
   :param image_inverse_transform: The inverse transform to apply on image tensor before displaying it.
                                   Default is None.
                                   For imagenet normalized image tensor, use deepml.transforms.ImageNetInverseTransform
   :param samples: The number of random image samples to display. Default is 9.
   :param cols: The number of display columns in the matplotlib figure. Default is 3.
   :param figsize: The matplotlib figure size. Default is (10,10)
   :param classes: The list of class names for class indices return by torch dataset.
   :param title_color: The title color for images.
   :return: None
   """
    x, y = get_random_samples_batch_from_dataset(dataset, samples=samples)
    x = transform_input(x, image_inverse_transform)

    if not classes and hasattr(dataset, 'classes'):
        classes = dataset.classes

    image_title_generator = ((x[index], transform_target(y[index], classes),
                              title_color) for index in range(x.shape[0]))
    plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)


def show_images_from_folder(img_dir, samples=9, cols=3, figsize=(10, 10), title_color=None):
    """
    Displays random samples of images from a folder.
    :param img_dir: The image directory containing images.
    :param samples: The number of random image samples to display. Default is 9.
    :param cols: The number of display columns in the matplotlib figure. Default is 3.
    :param figsize: The matplotlib figure size. Default is (10,10)
    :param title_color: The title color for images.
    :return: None
    """

    files = os.listdir(img_dir)
    if samples < len(files):
        samples = np.random.choice(files, size=samples, replace=False)
    else:
        samples = files

    image_generator = ((Image.open(os.path.join(img_dir, file)), file, title_color)
                       for file in samples)
    plot_images_with_title(image_generator, len(samples), cols=cols, figsize=figsize)


def show_images_from_dataframe(dataframe, img_dir=None, image_file_name_column="image",
                               label_column='label', samples=9, cols=3, figsize=(10, 10),
                               title_color=None):
    """
    Displays random samples of images from a dataframe using matplotlib figure.
    :param dataframe: The dataframe containing containing column for image filenames
    :param img_dir: The image directory. Default is None. If None, dataframe image filename column supposed
                    to contain full path to the image file.
    :param image_file_name_column: The name of the column containing image file names. Default is "image".
    :param label_column: The label columns containing the title for image file to be displayed. Default is 'label'.
    :param samples: The number of random image samples to display. Default is 9.
    :param cols: The number of display columns in the matplotlib figure. Default is 3.
    :param figsize: The matplotlib figure size. Default is (10,10)
    :param title_color: The title color for images.
    :return: None
    """
    samples = dataframe.sample(samples)
    image_generator = ((Image.open(os.path.join(img_dir, row_data[image_file_name_column])),
                        row_data[label_column], title_color)
                       for _, row_data in samples.iterrows())
    plot_images_with_title(image_generator, len(samples), cols=cols, figsize=figsize)
