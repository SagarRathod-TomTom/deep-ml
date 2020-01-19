
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageRowDataFrameDataset(Dataset):

    def __init__(self, X: pd.DataFrame, y: pd.Series, image_size=(28, 28),
                 transform=None):
        self.X_df = X
        self.y_df = y
        self.samples = X.shape[0]
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index):

        X = self.X_df.iloc[index]
        y = self.y_df.iloc[index]

        X = Image.fromarray(X.to_numpy().reshape(self.image_size).astype(np.uint8))
        if self.transform is not None:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return self.samples


class ImageFileDataFrameDataset(Dataset):

    def __init__(self, dataframe, img_file_path_column='image', target_column='target',
                 image_dir=None, transforms=None):

        self.dataframe = dataframe
        self.img_file_path_column = img_file_path_column
        self.target_column = target_column
        self.image_dir = image_dir
        self.transforms = transforms
        self.samples = self.dataframe.shape[0]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):

        image_file = self.dataframe.loc[index, self.img_file_path_column]

        if self.image_dir is not None:
            image_file = os.path.join(self.image_dir, image_file)

        X = Image.open(image_file)
        y = torch.tensor(self.dataframe.loc[index, self.target_column])

        if self.transforms is not None:
            X = self.transforms(X)

        return X, y


class SemSegImageFileDataFrameDataset(Dataset):

    def __init__(self, dataframe, image_dir, mask_dir, transforms=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.samples = self.dataframe.shape[0]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        image_file = self.dataframe.loc[index, 'image']

        image = Image.open(os.path.join(self.image_dir, image_file))
        mask = Image.open(os.path.join(self.mask_dir, image_file))

        if self.transforms is not None:
            image, mask = self.transforms(image=image, mask=mask)

        return image, mask