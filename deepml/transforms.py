import numpy as np
import torchvision
import torch
from deepml import constants


class AlbumentationTorchTranforms:
    """
        This class is a composition of albumentations augmentation and
        torchvision.transforms.ToTensor()
        This first applies albumentations transformations followed by
        torch transforms if any.

        albumentations transforms gets applied on both image and mask, however the
        torch transforms gets applied on only on input image and not on
        the target mask.
    """

    def __init__(self, albu_transforms=None, torch_transforms=None):
        super(AlbumentationTorchTranforms, self).__init__()
        self.albu_transforms = albu_transforms
        self.to_tensor = torchvision.transforms.ToTensor()
        self.torch_transforms = torch_transforms

    '''
    Accepts image and mask in python dict as PIL.Image or np.ndarray
    return torch tensor
    '''
    def __call__(self, image, mask):

        if type(image) != np.ndarray:
            image = np.array(image)

        if type(mask) != np.ndarray:
            mask = np.array(mask)

        if self.albu_transforms is not None:
            augmented = self.albu_transforms(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        if self.torch_transforms is not None:
            image = self.torch_transforms(image)

        if not isinstance(image, torch.Tensor):
            image = self.to_tensor(image)

        mask = torch.from_numpy(mask).astype(torch.FloatTensor)

        return image, mask


class ImageInverseTransform:
    """ Implementation of the inverse transform for image using mean and std_dev
        Accepts image_batch in #B, #C, #H #W order
    """
    def __init__(self, mean, std):
        super(ImageInverseTransform, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, image_batch):

        self.mean = self.mean.to(image_batch.device)
        self.std = self.std.to(image_batch.device)

        return image_batch * self.std[:, None, None] + self.mean[:, None, None]


class ImageNetInverseTransform(ImageInverseTransform):
    '''
       Imagenet inverse transform
       accepts image_batch in #B, #C, #H #W order
   '''
    def __init__(self):
        super(ImageNetInverseTransform, self).__init__(constants.IMAGENET_MEAN,
                                                       constants.IMAGENET_STD)
