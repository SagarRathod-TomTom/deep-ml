import os
import glob

import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from deepml.constants import RUN_DIR_NAME
from datetime import datetime


def get_datetime():
    date, timestamp = str(datetime.now()).split(' ')
    return '-'.join((date.replace('-', '_'), timestamp.replace(':', '_').split('.')[0]))


def find_new_run_dir_name(target_dir):
    files = glob.glob(os.path.join(target_dir, '{}*'.format(RUN_DIR_NAME)))

    if len(files) == 0:
        return RUN_DIR_NAME + str(1)

    run_numbers = map(lambda filename: int(filename.split('.')[-1]), files)

    # Return new run number
    return RUN_DIR_NAME + str(max(run_numbers) + 1)


def binarize(output: torch.FloatTensor, threshold: float = 0.50):
    output[output >= threshold] = 255
    output[output < threshold] = 0
    return output.to(torch.uint8)


def plot_images(image_title_generator, samples, cols=4, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    rows = int(np.ceil(samples / cols))
    for index, (image, title) in enumerate(image_title_generator):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])


def transform_target(target, classes=None):
    if target is not None:
        target = target.item() if type(target) == torch.Tensor else target
        if type(target) == float and classes is None:
            target = round(target, 2)
        elif type(classes) in (list, tuple) and classes:
            # if classes is not empty, replace target with actual class label
            target = classes[target]
    return target


def transform_input(X, image_inverse_transform=None):
    if image_inverse_transform is not None:
        X = image_inverse_transform(X.unsqueeze(dim=0)).squeeze()
    return X.numpy().transpose(1, 2, 0)  # CWH -> WHC


def show_images_from_loader(loader, image_inverse_transform=None, samples=9, cols=3, figsize=(5, 5),
                            classes=None):
    indexes = np.random.randint(0, len(loader.dataset), samples)

    def transform(input_batch):
        x, y = input_batch
        return transform_input(x, image_inverse_transform), transform_target(y, classes)

    image_title_generator = (transform(loader.dataset[index]) for index in indexes)
    plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)


def show_images_from_folder(img_dir, samples=9, cols=3, figsize=(10, 10)):
    files = os.listdir(img_dir)
    samples = np.random.randint(0, len(img_dir), size=samples)
    image_generator = ((Image.open(os.path.join(img_dir, files[index])), files[index])
                       for index in samples)
    plot_images(image_generator, len(samples), cols=cols, figsize=figsize)


def get_random_samples_batch_from_loader(loader):
    indexes = np.random.randint(0, len(loader.dataset), loader.batch_size)
    samples, targets = [], []
    for index in indexes:
        x, y = loader.dataset[index]
        samples.append(x)
        targets.append(y)
    return torch.stack(samples), torch.stack(targets)
