from typing import Union, Tuple, List, Callable, Any
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision


class ImageRowDataFrameDataset(torch.utils.data.Dataset):
    """ Class useful for reading images from a dataframe.
        Each row is assume to be the flattened array of an image.
        Each row is then reshaped to the provided image_size.
    """

    def __init__(self, dataframe: pd.DataFrame, target_column: str = None, image_size: Tuple[int, int] = (28, 28),
                 transform: torchvision.transforms = None):
        self.dataframe = dataframe.reset_index(drop=True, inplace=False)
        self.target_column = None

        if target_column:
            self.target_column = self.dataframe[target_column]
            self.dataframe.drop(target_column, axis=1, inplace=True)

        self.samples = self.dataframe.shape[0]
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index: int):

        X = self.dataframe.iloc[index]

        X = Image.fromarray(X.to_numpy().reshape(self.image_size).astype(np.uint8))
        if self.transform is not None:
            X = self.transform(X)

        y = 0
        if self.target_column is not None:
            y = self.target_column.loc[index]

        return X, y

    def __len__(self):
        return self.samples


class ImageDataFrameDataset(torch.utils.data.Dataset):
    """ This class is useful for reading dataset of images for image classification/regression problem.
    """

    def __init__(self, dataframe: pd.DataFrame, image_file_name_column: str = 'image', target_columns: Union[int, List[
        str]] = None, image_dir: str = None, transforms: Union[torchvision.transforms, Callable] = None,
                 target_transform: Union[torchvision.transforms, Callable] = None,
                 open_file_func: Callable[[Any], np.ndarray] = None):

        self.dataframe = dataframe.reset_index(drop=True, inplace=False)
        self.image_file_name_column = image_file_name_column
        self.target_columns = target_columns
        self.image_dir = image_dir
        self.transforms = transforms
        self.samples = self.dataframe.shape[0]
        self.target_transform = target_transform
        self.open_file_func = open_file_func

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:

        image_file = self.dataframe.loc[index, self.image_file_name_column]

        if self.image_dir:
            image_file = os.path.join(self.image_dir, image_file)

        if self.open_file_func is None:
            X = Image.open(image_file)
        else:
            X = self.open_file_func(image_file)

        if self.transforms is not None:
            X = self.transforms(X)

        y = 0
        if self.target_columns:
            y = torch.tensor(self.dataframe.loc[index, self.target_columns])
            if self.target_transform:
                y = self.target_transform(y)

        return X, y


class ImageListDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir: str, transforms: torchvision.transforms = None,
                 open_file_func: Callable[[Any], Union[np.ndarray, Image]] = None):

        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transforms = transforms
        self.open_file_func = open_file_func

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):

        image_file = self.images[index]
        if self.open_file_func is None:
            X = Image.open(os.path.join(self.image_dir, image_file))
        else:
            X = self.open_file_func(os.path.join(self.image_dir, image_file))

        if self.transforms is not None:
            X = self.transforms(X)

        return X, image_file


class SegmentationDataFrameDataset(torch.utils.data.Dataset):
    """
        This class is useful for reading images and mask labels required for
        semantic segmentation problems.

        The image file and corresponding mask label file should have the same name,
        and should be stored in a provided mask directory.

        You can provide custom open_file_func for reading image, by default it uses PIL Image.open()
        open_file_func should accept 2 parameters: image_file_path, mask_file path
        and return image, mask
    """

    def __init__(self, dataframe: pd.DataFrame, image_dir: str, mask_dir: str = None, image_col: str = 'image',
                 mask_col: str = None, albu_torch_transforms=None,
                 target_transform=None, train: bool = True,
                 open_file_func: Callable[[Any], Union[np.ndarray, Image]] = None):
        """

        :param dataframe: the pandas dataframe
        :param image_dir: the dir path containing training images
        :param mask_dir: the dir path containing mask images
        :param image_col: the name of column in dataframe, file is fetched
                          by path joining os.path.join(image_dir, df.loc[index, image_col]
        :param mask_col:  Same as image_col, If None image_col's filename is used for mask.
        :param albu_torch_transforms: albumentation transforms for both image and target mask
        :param target_transform: transform for only target mask for preprocessing.
        :param train: If true, returns tuple of tensors else return image tensor with filename. Default is True.
        :param open_file_func:  callable function to open image and mask file. By default PIL.Image.open is used.
        """
        if train:
            assert mask_dir, "For training purpose, mask_dir should not be None"

        self.dataframe = dataframe.reset_index(drop=True, inplace=False)
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_col = image_col
        self.mask_col = mask_col if mask_col else image_col

        self.albu_torch_transforms = albu_torch_transforms
        self.target_transform = target_transform
        self.samples = self.dataframe.shape[0]
        self.train = train
        self.open_file_func = open_file_func

    def __len__(self):
        return self.samples

    def __getitem__(self, index):

        image_file = os.path.join(self.image_dir, self.dataframe.loc[index, self.image_col])
        mask_file = os.path.join(self.mask_dir, self.dataframe.loc[index, self.mask_col]) if self.train else None

        if self.open_file_func is None:
            image = np.array(Image.open(image_file))
            mask = np.array(Image.open(mask_file)) if self.train else None
        else:
            image = self.open_file_func(image_file)
            mask = self.open_file_func(mask_file) if self.train else None

        if self.train:
            transformed = self.albu_torch_transforms(image=image, mask=mask)
        else:
            transformed = self.albu_torch_transforms(image=image)

        if self.train and self.target_transform:
            transformed['mask'] = self.target_transform(transformed['mask'])

        if self.train:
            return transformed['image'], transformed['mask']
        else:
            return transformed['image'], self.dataframe.loc[index, self.image_col]
