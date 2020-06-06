import os
import glob

import torch
from matplotlib import pyplot as plt
import numpy as np
from deepml.constants import RUN_DIR_NAME


def find_current_run_number(target_dir):
    files = glob.glob(os.path.join(target_dir, '{}*'.format(RUN_DIR_NAME)))

    if len(files) == 0:
        return 1

    run = 0
    for file in files:
        current = int(os.path.split(file)[1].split('_')[1])
        if current > run:
            run = current

    # Return new run number
    return run + 1


def binarize(output: torch.FloatTensor, threshold: float = 0.50):
    output[output >= threshold] = 255
    output[output < threshold] = 0
    return output.to(torch.uint8)


def show_batch(loader, cols=4, figsize=(5, 5)):
    samples = np.random.randint(0, len(loader.dataset), loader.batch_size)
    plt.figure(figsize=figsize)
    rows = int(np.ceil(loader.batch_size / cols))
    for index, sample_index in enumerate(samples):
        X, y = loader.dataset[sample_index]
        X = X.numpy().transpose(1, 2, 0) # CWH -> WHC
        plt.subplot(rows, cols, index + 1)
        plt.imshow(X)
        if y is not None:
            target = y.item()
            title = round(target, 2) if type(target) == float else target
            plt.title(title)
        plt.xticks([])
        plt.yticks([])


def plot_image_batch(img_batch: torch.Tensor, targets=None, cols=4, figsize=(5, 5)):
    img_batch = img_batch.numpy().transpose((0, 2, 3, 1)) # BCWH --> BWHC
    plt.figure(figsize=figsize)
    samples = len(img_batch)
    rows = int(np.ceil(samples / cols))
    for index in range(samples):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(img_batch[index])
        if targets is not None:
            target = targets[index].item()
            title = round(target, 2) if type(target) == float else target
            plt.title(title)
        plt.xticks([])
        plt.yticks([])