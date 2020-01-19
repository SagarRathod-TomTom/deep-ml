import numpy as np
import torchvision


class AlbumentationTorchCompat(object):

    def __init__(self, albu_transforms=None, torch_transforms=None, apply_torch_transforms_to_mask=False):
        super(AlbumentationTorchCompat, self).__init__()
        self.albu_transforms = albu_transforms
        self.to_tensor = torchvision.transforms.ToTensor()
        self.torch_transforms = torch_transforms
        self.apply_torch_transforms_to_mask = apply_torch_transforms_to_mask

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

        image = self.to_tensor(image)

        if self.apply_torch_transforms_to_mask:
            mask = self.to_tensor(mask)

        if self.torch_transforms is not None:
            image = self.torch_transforms(image)

        return image, mask