import os
import glob

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deepml.constants import RUN_DIR_NAME
from datetime import datetime
import pkg_resources

font_resource = pkg_resources.resource_filename(__name__, "resources/fonts/OpenSans-Light.ttf")
FONT = ImageFont.truetype(font_resource, 16)


def create_text_image(text, img_size=(224, 224), text_color='black'):
    image = Image.new('RGB', img_size, color=(255, 255, 255))
    img_width, img_height = img_size
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textsize(text, font=FONT)
    draw.text(((img_width - text_width)/2, (img_height - text_height)/2), text, fill=text_color,
              align='center', font=FONT)
    return image


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


def transform_target(target, classes=None):
    """
    Accepts target value either single dimensional torch.Tensor or (int, float)
    :param target:
    :param classes:
    :return:
    """
    if target is not None:
        target = target.item() if type(target) == torch.Tensor else target
        if type(target) == float and classes is None:
            target = round(target, 2)
        elif type(classes) in (list, tuple) and classes:
            # if classes is not empty, replace target with actual class label
            target = classes[target]
    return target


def transform_input(x, image_inverse_transform=None):
    """
    Accepts input image batch in #BCHW form

    :param x: input image batch
    :param image_inverse_transform: an optional inverse transform to apply
    :return:
    """
    if image_inverse_transform is not None:
        x = image_inverse_transform(x)

    # #BCHW --> #BHWC
    return x.permute([0, 2, 3, 1])


def get_random_samples_batch_from_loader(loader, samples=None):

    if len(loader) == 0:
        raise ValueError('Loader is empty')
    return get_random_samples_batch_from_dataset(loader.dataset,
                                                 loader.batch_size if samples is None else samples)


def get_random_samples_batch_from_dataset(dataset, samples=8):

    if len(dataset) == 0:
        raise ValueError('Dataset is empty')

    indexes = np.random.randint(0, len(dataset), samples)
    samples, targets = [], []
    for index in indexes:
        x, y = dataset[index]
        samples.append(x)
        targets.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    return torch.stack(samples), torch.stack(targets)
