import torch
import random
import numpy as np


class Compose:
    """
    Compose a list of pose transforms
    
    Args:
        transforms (list): List of transforms to be applied.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: dict):
        """Applies the given list of transforms

        Args:
            x (dict): input data

        Returns:
            dict: data after the transforms
        """
        for transform in self.transforms:
            x = transform(x)
        return x


# Adopted from: https://github.com/AmitMY/pose-format/
class ShearTransform:
    """
    Applies `2D shear <https://en.wikipedia.org/wiki/Shear_matrix>`_ transformation
    
    Args:
        shear_std (float): std to use for shear transformation. Default: 0.2
    """
    def __init__(self, shear_std: float=0.2):
        self.shear_std = shear_std

    def __call__(self, data:dict):
        """
        Applies shear transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after shear transformation
        """
        
        x = data
        assert x.shape[0] == 2, "Only 2 channels inputs supported for ShearTransform"
        x = x.permute(1, 2, 0) #CTV->TVC
        shear_matrix = torch.eye(2)
        shear_matrix[0][1] = torch.tensor(
            np.random.normal(loc=0, scale=self.shear_std, size=1)[0])
        res = torch.matmul(x.float(), shear_matrix.float())
        data = res.permute(2, 0, 1) #TVC->CTV
        return data.double()


class RotatationTransform:
    """
    Applies `2D rotation <https://en.wikipedia.org/wiki/Rotation_matrix>`_ transformation.
    
    Args:
        rotation_std (float): std to use for rotation transformation. Default: 0.2
    """
    def __init__(self, rotation_std: float=0.2):
        self.rotation_std = rotation_std

    def __call__(self, data):
        """
        Applies rotation transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after rotation transformation
        """
        x = data
        assert x.shape[0] == 2, "Only 2 channels inputs supported for RotationTransform"
        x = x.permute(1, 2, 0) #CTV->TVC
        rotation_angle = torch.tensor(
            np.random.normal(loc=0, scale=self.rotation_std, size=1)[0]
        )
        rotation_cos = torch.cos(rotation_angle)
        rotation_sin = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor(
            [[rotation_cos, -rotation_sin], [rotation_sin, rotation_cos]],
            dtype=torch.float32,
        )
        res = torch.matmul(x.float(), rotation_matrix.float())
        data = res.permute(2, 0, 1) #TVC->CTV
        return data.double()



