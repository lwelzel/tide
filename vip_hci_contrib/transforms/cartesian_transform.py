"""
This module provides a class for transforming ASDI data cubes.

Author: Lukas Welzel
Date: 19.10.2023

Copyright (C) 2023 Lukas Welzel
"""

__author__ = "L. Welzel"
# __all__ = []


from typing import Tuple, Union
from abc import ABC, ABCMeta


import torch
import kornia


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CubeTransformMeta():
    @staticmethod
    def scale(tensor: torch.Tensor, scale_factors: torch.Tensor) -> torch.Tensor:
        scaled_cube = kornia.geometry.transform.scale(
            tensor,
            scale_factor=scale_factors,
            mode="bilinear"
        )
        return scaled_cube

    @staticmethod
    def rescale(tensor: torch.Tensor, scale_factors: torch.Tensor) -> torch.Tensor:
        rescaled_cube = kornia.geometry.transform.scale(
            tensor=tensor,
            scale_factor=1. / scale_factors,
            mode="bilinear"
        )
        return rescaled_cube

    @staticmethod
    def derotate(tensor: torch.Tensor, derot_angles: torch.Tensor) -> torch.Tensor:
        derot_cube = kornia.geometry.transform.rotate(
            torch.moveaxis(tensor, 1, 0),
            - derot_angles,  # negative!
            mode="bilinear"
        )
        derot_cube = torch.moveaxis(derot_cube, 0, 1)
        return derot_cube

    @staticmethod
    def rotate(tensor: torch.Tensor, derot_angles: torch.Tensor) -> torch.Tensor:
        rot_cube = kornia.geometry.transform.rotate(
            torch.moveaxis(tensor, 1, 0),
            derot_angles,
            mode="bilinear"
        )
        rot_cube = torch.moveaxis(rot_cube, 0, 1)
        return rot_cube

    @staticmethod
    def translate(tensor: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        trans_cube = kornia.geometry.transform.translate(
            torch.moveaxis(tensor, 1, 0),
            translation=translation,
            mode="bilinear"
        )
        trans_cube = torch.moveaxis(trans_cube, 0, 1)
        return trans_cube

    def rescale_derotate(self, tensor: torch.Tensor,
                         scale_factors: torch.Tensor, derot_angles: torch.Tensor) -> torch.Tensor:
        rescaled_cube = self.rescale(tensor, scale_factors=scale_factors)
        derot_cube = self.derotate(rescaled_cube, derot_angles=derot_angles)
        return derot_cube

    def rotate_scale(self, tensor: torch.Tensor,
                     scale_factors: torch.Tensor, derot_angles: torch.Tensor) -> torch.Tensor:
        rot_cube = self.rotate(tensor, derot_angles=derot_angles)
        scaled_cube = self.scale(rot_cube, scale_factors=scale_factors)
        return scaled_cube

    @staticmethod
    def _calculate_center(tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Calculate the center of a 2D plane in a tensor.

        This function assumes the tensor represents an image with dimensions (channels, parallactic_angles, x, y)
        and calculates the center of the (x, y) plane. The calculation adheres to the conventions for handling
        odd- and even-dimension images, ensuring the center is placed on a pixel for odd dimensions and between
        pixels for even dimensions.

        Args:
            tensor (Tensor): A 4D tensor from which the size of the 2D plane (x, y dimensions) is extracted.

        Returns:
            Tuple[int, int]: Center coordinates (cy, cx) based on zero-based indexing.
        """

        __, __, X, Y = tensor.shape

        cy, cx = X / 2, Y / 2
        if X % 2:
            cy -= 0.5
        if Y % 2:
            cx -= 0.5
        return int(cy), int(cx)

    @staticmethod
    def cart_to_polar(y, x):
        rho = torch.sqrt(y ** 2 + x ** 2)
        phi = torch.rad2deg(torch.atan2(y, x))
        return rho, phi

    @staticmethod
    def polar_to_cart(rho, phi):
        phi_rad = torch.deg2rad(phi)

        x = rho * torch.cos(phi_rad)
        y = rho * torch.sin(phi_rad)

        return x, y

    def cart_to_polar_relative(self, tensor: torch.Tensor, y: Union[torch.Tensor, float, int], x: Union[torch.Tensor, float, int]) -> Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts Cartesian coordinates to polar coordinates with respect to the center of the image represented by the tensor.

        Args:
            tensor (Tensor): A 4D tensor representing an image, used to determine the center of the 2D plane.
            y (int or float): The Y coordinate in Cartesian space, relative to the array's zero index.
            x (int or float): The X coordinate in Cartesian space, relative to the array's zero index.

        Returns:
            Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]]:
            The polar coordinates (rho, phi), where rho is the radius and phi is the angle in degrees.
        """
        cy, cx = self._calculate_center(tensor)
        # Adjust coordinates to be relative to the image center
        y_centered = y - cy
        x_centered = x - cx
        rho = torch.sqrt(y_centered ** 2 + x_centered ** 2)
        phi = torch.rad2deg(torch.atan2(y_centered, x_centered))
        return rho, phi

    def polar_to_cart_relative(self, tensor: torch.Tensor, rho: Union[torch.Tensor, float, int], phi: Union[torch.Tensor, float, int]) -> Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts polar coordinates to Cartesian coordinates with respect to the center of the image represented by the tensor.

        Args:
            tensor (Tensor): A 4D tensor representing an image, used to determine the center of the 2D plane.
            rho (float): The radius in polar coordinates.
            phi (float): The angle in degrees in polar coordinates.

        Returns:
            Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]]:
            The Cartesian coordinates (y, x), relative to the array's zero index.
        """
        cy, cx = self._calculate_center(tensor)
        phi_rad = torch.deg2rad(phi)
        # Calculate Cartesian coordinates relative to the image center
        x_centered = rho * torch.cos(phi_rad)
        y_centered = rho * torch.sin(phi_rad)
        # Adjust coordinates to be relative to the array's zero index
        y = y_centered + cy
        x = x_centered + cx
        return y, x

    def circular_mask_2d_scaled(self, tensor: torch.Tensor, fwhm: float, n: float = 1.) -> torch.Tensor:
        C, T, X, Y = tensor.shape
        cy, cx = self._calculate_center(tensor)

        radius = n * fwhm

        # Create a grid for X and Y
        x = torch.arange(X).unsqueeze(1).expand(X, Y) - cx
        y = torch.arange(Y).unsqueeze(0).expand(X, Y) - cy

        # Calculate squared distance from the center
        squared_dist = x ** 2 + y ** 2

        # Generate the circular mask
        circular_mask = squared_dist >= radius ** 2

        return circular_mask

    def circular_mask_2d_custom_center(self, tensor: torch.Tensor, x: int, y: int, radius: float) -> torch.Tensor:
        """
        Create a circular mask for a 4D tensor with a custom center and radius.

        Args:
        tensor (torch.Tensor): Input tensor of shape (C, T, X, Y).
        x (int): X-coordinate of the custom center.
        y (int): Y-coordinate of the custom center.
        radius (float): Radius of the circle.

        Returns:
        torch.Tensor: Mask tensor of the same shape as input, with 1s outside the circle and 0s inside.
        """
        C, T, X, Y = tensor.shape
        cy, cx = self._calculate_center(tensor)

        # Create a grid for X and Y
        x_grid = torch.arange(X).unsqueeze(0).expand(X, Y) - cx - x
        y_grid = torch.arange(Y).unsqueeze(1).expand(X, Y) - cy - y

        # Calculate squared distance from the custom center
        squared_dist = x_grid ** 2 + y_grid ** 2

        # Generate the circular mask
        circular_mask = ~torch.ge(squared_dist, radius ** 2)

        return circular_mask


if __name__ == "__main__":
    cartesian_transform = CubeTransformMeta()

    tensor = torch.ones(5, 5, 189, 189)

    mask = cartesian_transform.circular_mask_2d_custom_center(tensor, 30, 30, 20)
    print(mask.shape)

    import matplotlib.pyplot as plt

    im = plt.imshow(mask.cpu().numpy(), origin="lower")
    plt.colorbar(im)
    plt.show()
