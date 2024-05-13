"""
CODI Tensor Decomposition Module in Polar Coordinates

This module provides classes for CODI by tensor decomposition using various algorithms.

Author: Lukas Welzel
Date: 19.10.2023

Copyright (C) 2023 Lukas Welzel
"""

__author__ = "L. Welzel"
# __all__ = []


from typing import Type, Any, List, Optional, Tuple, Callable, NoReturn, Union
from dataclasses import dataclass
from abc import ABCMeta, ABC, abstractmethod
from functools import partial

# from vip_hci_contrib.transforms.polar_transform import PolarTensorTransform, WarpPlaceholder  # previous implementation
from vip_hci_contrib.transforms.polar_transform import PolarTransformModule
from vip_hci_contrib.decompositions.decomposition_codi_base import AbstractBaseTensorDecompositionCODI
from vip_hci_contrib.decompositions.decomposition_bases import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tl.set_backend("pytorch")


class AbstractBasePolarTensorDecompositionCODI(AbstractBaseTensorDecompositionCODI, ABC):
    default_polar_warp_settings = {
        "initialRadius": 7.,
        "finalRadius": 20.,
        "initialAngle": None,
        "finalAngle": None
    }

    def __init__(self, cube: torch.Tensor, angle_list: torch.Tensor, scale_list: torch.Tensor, fwhm: torch.Tensor = None,
                 build_transform=True, backend: str = "tensorly",
                 **kwargs):

        super(AbstractBasePolarTensorDecompositionCODI, self).__init__(cube=cube,
                                                                       angle_list=angle_list, scale_list=scale_list,
                                                                       fwhm=fwhm,
                                                                       backend=backend,
                                                                       **kwargs)

        # TODO: kwargs should be split
        self.polar_warp_settings = {**self.__class__.default_polar_warp_settings, **kwargs}

        if build_transform:
            self.polar_transform = PolarTransformModule(tensor=cube)


    def forward(self, cube: torch.Tensor, angle_list: Any = None, scale_list: Any = None, fwhm: Any = None, verbose: bool = False, psf: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # option one:
        # TODO: rescaling doesnt work properly right now since it does not account for the missing radius
        # cube = self.warp.to_polar(cube)
        # residual = self._base_forward(cube, angle_list=angle_list, fwhm=fwhm, verbose=verbose, **kwargs)
        # residual = self.warp.to_cart(residual)

        # option two:
        # Prioritize settings passed during method call over stored ones
        method_settings = {**self.psf_settings, **kwargs}

        # CDI SCALING
        scaled_cube = self.scale(cube, scale_factors=self.scale_factors)
        scaled_cube = scaled_cube - torch.median(scaled_cube, keepdim=True, dim=1).values
        # scaled_cube = scaled_cube - torch.median(scaled_cube, keepdim=True, dim=0).values
        del cube

        scaled_cube = self.polar_transform.cart2pol(scaled_cube)

        # PSF model
        psf_model = self.psf_func(scaled_cube, **method_settings)

        # residuals
        residual = scaled_cube - psf_model
        del scaled_cube
        del psf_model

        residual = self.polar_transform.pol2cart(residual)

        # we cant use _base_rescale_derotate because self contains the polar versions of rescale and derotate,
        # so when _base_rescale_derotate calls them it calls the polar versions (fused with partial)
        # residual = self._base_rescale_derotate(residual)
        residual = self.rescale(residual, scale_factors=self.scale_factors)
        residual = self.derotate(residual, derot_angles=self.derot_angles)

        return residual

    def get_parameters(self):
        return {**self.psf_settings, **self.polar_warp_settings}

    @staticmethod
    def circular_mask(*args, **kwargs):
        """
        Polar decompositions dont mask at the moment.
        # TODO: include support for masking of polar decompositions instead of the weird warp workaround
        """
        return None


if __name__ == "__main__":
    pass
