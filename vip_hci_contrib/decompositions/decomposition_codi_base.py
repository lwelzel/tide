"""
CODI Tensor Decomposition Module

This module provides classes for CODI by tensor decomposition using various algorithms.

Author: Lukas Welzel
Date: 19.10.2023

Copyright (C) 2023 Lukas Welzel
"""

__author__ = "L. Welzel"
# __all__ = []

import sys

import matplotlib.pyplot as plt
import numpy as np
from typing import Type, Any, List, Optional, Tuple, Callable, NoReturn, Union
from dataclasses import dataclass
from abc import ABC
from functools import partial

import torch
import kornia

import tensorly as tl
import tensorly.decomposition as tld
import tntorch as tn

from vip_hci_contrib.transforms.cartesian_transform import CubeTransformMeta


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tl.set_backend("pytorch")
# TODO: potentially use: torch.inference_mode instead of no_grad
# https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099?u=timgianitsos
torch.no_grad


@dataclass
class TensorDecomposition_Params:
    """
    Set of parameters for the Tensor Decomposition algorithms.
    See tensor decomposition functions below for the documentation.
    """
    cube: np.ndarray = None
    angle_list: np.ndarray = None
    fwhm: float = None
    full_output: bool = False
    verbose: bool = True
    debug: bool = False

    @staticmethod
    def create_for_decomposition(decomposition_name: str):
        class_name = f"{decomposition_name.upper()}_Params"

        # Get attributes of TensorDecomposition_Params and convert to dictionary
        attributes = dict(vars(TensorDecomposition_Params))

        # Create a new dataclass that inherits from TensorDecomposition_Params
        new_class = type(class_name, (TensorDecomposition_Params,), attributes)
        new_class.__module__ = "vip_hci_contrib.decompositions"  # Set the module attribute

        # Add the new_class to the 'vip_hci_contrib.decompositions' module
        decompositions_module = sys.modules['vip_hci_contrib.decompositions']
        setattr(decompositions_module, class_name, new_class)

        return new_class


class AbstractBaseTensorDecompositionCODI(torch.nn.Module, CubeTransformMeta, ABC):
    """
    Abstract base class for CODI (ASDI) tensor decomposition.

    Args:
        derot_angles (torch.Tensor): Tensor containing derotation angles.
        lbda (torch.Tensor): Lambda values.
        backend (str, optional): Backend for decomposition (default is "tensorly").
        **kwargs: Additional keyword arguments for decomposition.

    Attributes:
        default_psf_settings (dict): Default settings for the decompositions.
        derot_angles (torch.Tensor): Tensor containing derotation angles.
        lbda (torch.Tensor): Lambda values.
        scale_factors (torch.Tensor): Scaling factors for CDI.
        psf_settings (dict): Settings for the Point Spread Function (PSF) model.
        backend (str): Backend for decomposition.
        psf_func (Callable): Function for PSF modeling.

    """

    # default settings for the decompositions
    default_psf_settings = {}

    def __init__(self, cube: torch.Tensor, angle_list: torch.Tensor, scale_list: torch.Tensor, fwhm: torch.Tensor = None,
                 backend: str = "tensorly",
                 **kwargs):
        super(AbstractBaseTensorDecompositionCODI, self).__init__()

        self.cube_shape = cube.shape

        self.derot_angles = angle_list
        self.lbda = scale_list

        self.scale_factors = torch.tile(self.lbda.max() / self.lbda, (2, 1)).T  # kornia is weird like that

        # Merge class-specific defaults with provided settings
        self.psf_settings = {**self.__class__.default_psf_settings, **kwargs}

        # Store which backend to use
        self.backend = backend

        # setup for center masking
        self.fwhm = fwhm
        self.mask = None

        # errors tracking
        self.rec_errors = []

        # Determine which function to use
        if backend == "tensorly":
            method_name = "_psf_model_tensorly"
        elif backend == "tntorch":
            method_name = "_psf_model_tntorch"
        else:
            method_name = "_psf_model"  # default method name

        # TODO: should be properly defined, not in init block
        self.psf_func = getattr(self, method_name, self._psf_model)

        self.__name__ = self.__class__.__name__

        self.detection_module = STIMModule(self.derot_angles)

        self.total_number_parameters = -1

    def forward(self, cube: torch.Tensor, angle_list: Any = None, scale_list: Any = None, fwhm: Any = None, verbose: bool = False, psf: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # Prioritize settings passed during method call over stored ones
        method_settings = {**self.psf_settings, **kwargs}

        # if (cube.shape[1] > 2) and (abs(self.derot_angles[-1] - self.derot_angles[0]) > 5.):
        #     cube = cube - torch.median(cube, keepdim=True, dim=1).values

        mask_center = kwargs.pop("mask", True)
        if mask_center:
            rad = torch.quantile(self.fwhm, q=0.1).item()
            self.mask = self.circular_mask_2d_scaled(cube, rad, n=0.5).view(1, 1, *cube.shape[-2:])

        # CDI SCALING
        scaled_cube = self.scale(cube, scale_factors=self.scale_factors) * self.mask
        scaled_cube = scaled_cube.contiguous()

        # PSF model
        psf_model = self.psf_func(scaled_cube, **method_settings)

        # Residuals
        residual = scaled_cube - psf_model

        residual = self.rescale_derotate(residual, scale_factors=self.scale_factors, derot_angles=self.derot_angles)

        # residual = self.matched_filter(residual, psf)
        # residual = self.gaussian_filter(residual)

        return residual

    def vip_wrapper(self, cube: np.ndarray, angle_list: np.ndarray = None, scale_list: np.ndarray = None, fwhm: Any = None, verbose: bool = False, psf: np.ndarray = None, **kwargs):
        cube = torch.tensor(cube, dtype=torch.float32).to(device)
        angle_list = torch.tensor(angle_list, dtype=torch.float32).to(device)
        scale_list = torch.tensor(scale_list, dtype=torch.float32).to(device)
        fwhm = torch.tensor(fwhm, dtype=torch.float32).to(device)
        psf = psf # torch.tensor(psf, dtype=torch.float32).to(device)

        residual = self.forward(cube, angle_list=angle_list, scale_list=scale_list, fwhm=fwhm, verbose=verbose, psf=psf, **kwargs)

        residual, __ = self.detection_module(residual)

        return residual.cpu().detach().numpy()


    def matched_filter(self, tensor: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        # using 2D convolutions per channel
        mf_tensor = kornia.filters.filter2d(tensor, psf, normalized=True)

        return mf_tensor

    def gaussian_filter(self, tensor: torch.Tensor, fwhm: float = 5.) -> torch.Tensor:
        sigma = fwhm * 0.42466090014400953  # FWHM to sigma
        mf_tensor = kornia.filters.gaussian_blur2d(tensor, kernel_size=29, sigma=(sigma, sigma))

        return mf_tensor

    def get_parameters(self):
        return {**self.psf_settings}

    def circular_mask(self, tensor, fwhm, n=2):
        """
        Create a circular mask for a 4D tensor (C, T, X, Y).

        Args:
        tensor (torch.Tensor): Input tensor of shape (C, T, X, Y).
        fwhm (float): Full-width half-maximum.
        n (float): Multiple of FWHM to determine the radius of the circle.

        Returns:
        torch.Tensor: Mask tensor of the same shape as input.
        """

        if fwhm is None:
            return None

        C, T, X, Y = tensor.shape
        if isinstance(fwhm, torch.Tensor):
            # fwhm = torch.mean(fwhm)
            fwhm = torch.quantile(fwhm, q=0.2)  # quantile because there can be some quite large outliers

        circular_mask = self.circular_mask_2d_scaled(tensor, fwhm, n=n)

        # Expand mask to match the shape of the tensor
        mask = circular_mask.unsqueeze(0).unsqueeze(0).expand(C, T, X, Y)

        return mask.float()

    @property
    def reconstruction_errors(self):
        if len(self.rec_errors) == 0:
            return None
        max_len = max(len(lst) for lst in self.rec_errors)

        reconstruction_errors = np.full((len(self.rec_errors), max_len), -1.0, dtype=float)

        for i, lst in enumerate(self.rec_errors):
            for j, tensor in enumerate(lst):
                # Convert the tensor to a CPU tensor and then to a NumPy float
                reconstruction_errors[i, j] = tensor.cpu().item()

        return reconstruction_errors

    def __repr__(self):
        # Creating a string representation of the derot_angles and lbda Tensors
        derot_angles_repr = (f"min: {self.derot_angles.min():.0f},"
                             f" max: {self.derot_angles.max():.0f},"
                             f" shape: {self.derot_angles.shape}")
        lbda_repr = (f"min: {self.lbda.min()},"
                     f" max: {self.lbda.max()},"
                     f" shape: {self.lbda.shape}")

        # Creating the PSF settings string
        psf_settings_str = "\n\t".join([f"  {k}: {v}" for k, v in self.psf_settings.items()])

        return (f"\n{self.__class__.__name__}\n"
                f"  derot_angles: {derot_angles_repr},\n"
                f"  lbda:         {lbda_repr},\n"
                f"  backend:      {self.backend!r},\n"
                f"  PSF settings:\n"
                f"\t{psf_settings_str}\n")


class STIMModule(torch.nn.Module, CubeTransformMeta):
    def __init__(self, derot_angles: torch.Tensor):
        super(STIMModule, self).__init__()

        self.derot_angles = derot_angles

        self._vec_forward = torch.vmap(self.adi_forward, in_dims=0, out_dims=0)

    def forward(self, tensor: torch.Tensor, collapse=True, **kwargs) -> (torch.Tensor, torch.Tensor):
        det_map = self._vec_forward(tensor, **kwargs)
        inv_map = self.inverse_stim_map(tensor)

        if collapse:
            det_map = torch.nanmedian(det_map, dim=0).values

        inv_map = torch.max(inv_map, dim=0).values

        norm_det_map = det_map / inv_map
        try:
            norm_fac = torch.abs(torch.max(norm_det_map[~torch.isnan(norm_det_map)]))
        except RuntimeError as e:
            print(e, e.args, e.__cause__)
            print("Not normalizing STIM (custom)")
            norm_fac = 1.
        norm_det_map = norm_det_map / norm_fac

        try:
            det_map = det_map / torch.abs(torch.max(det_map[~torch.isnan(det_map)]))
        except RuntimeError as e:
            print(e, e.args, e.__cause__)
            print("Not normalizing STIM")

        return det_map, norm_det_map

    @staticmethod
    def adi_forward(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if tensor.ndim == 4:
            tensor = torch.mean(tensor, dim=0)

        t, n, _ = tensor.shape
        mu = torch.mean(tensor, dim=0)
        sigma = torch.sqrt(torch.var(tensor, dim=0))

        detection_map = torch.divide(mu, sigma)

        sy, sx = detection_map.shape

        cy, cx = sy / 2, sx / 2
        if sx % 2:
            cy -= 0.5
        if sy % 2:
            cx -= 0.5

        radius = int(np.round(n/2.))

        yy = torch.linspace(0, sy, sy, dtype=torch.int64).unsqueeze(1)
        xx = torch.linspace(0, sx, sx, dtype=torch.int64).unsqueeze(0)

        circle = (yy - cy) ** 2 + (xx - cx) ** 2

        circle_mask = torch.lt(circle, radius ** 2)

        return detection_map * circle_mask

    def inverse_stim_map(self, tensor: torch.Tensor) -> torch.Tensor:
        counter_rotated_cube = self.derotate(tensor, -2. * self.derot_angles)
        inverse_stim_map = self._vec_forward(counter_rotated_cube)

        return inverse_stim_map


def vip_torch_numpy_wrapper(DecompositionClass: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    A wrapper class to adapt PyTorch-based tensor decomposition classes
    to handle NumPy arrays while utilizing GPU acceleration when available.

    The wrapped class will:
    - Automatically convert input NumPy arrays to torch tensors and move them to the GPU.
    - Process the tensors using the specified tensor decomposition class.
    - Convert the torch tensor outputs back to NumPy arrays and move them to the CPU.

    Parameters:
    - DecompClass: A tensor decomposition class based on PyTorch.

    Returns:
    - Wrapped class with enhanced functionality to handle NumPy arrays and GPU acceleration.
    """

    class VIPDecompositionWrapper(DecompositionClass):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Check if CUDA (GPU support for PyTorch) is available and set the device accordingly
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.return_full = kwargs.pop("return_full", False)

            # Convert numpy arrays in args to torch tensors
            args = [torch.tensor(arg, dtype=torch.float32).to(self.device) if isinstance(arg, np.ndarray) else arg for
                    arg in args]

            # Convert numpy arrays in kwargs to torch tensors
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    kwargs[key] = torch.tensor(value, dtype=torch.float32).to(self.device)

            super(VIPDecompositionWrapper, self).__init__(*args, **kwargs)

            # TODO: this is so horrible, fix later
            if self.return_full:
                self.forward = self.forward_full
            else:
                self.forward = self.forward_collapsed

            self.to(self.device)  # Move the entire model to the chosen device


        def _forward(self, cube: np.ndarray, angle_list: Any = None, verbose: bool = False, **kwargs: Any) -> torch.Tensor:
            """
            Overrides the forward method of the decomposition class to handle NumPy arrays
            and to utilize GPU acceleration.

            Parameters:
            - cube: NumPy array containing the data to be processed.
            - angle_list: List of angles.
            - verbose: Verbosity flag.
            - kwargs: Additional keyword arguments to be passed to the superclass's forward method.

            Returns:
            - NumPy array containing the processed data.
            """

            # Step 1: Convert the numpy cube to a torch tensor and send to the chosen device (e.g., CUDA GPU)
            cube_torch = torch.tensor(cube, dtype=torch.float32).to(self.device)

            if kwargs.get("fwhm") is not None:
                if isinstance(kwargs["fwhm"], np.ndarray):
                    kwargs["fwhm"] = torch.tensor(kwargs["fwhm"], dtype=torch.float32).to(self.device)
                elif isinstance(kwargs["fwhm"], float):
                    kwargs["fwhm"] = torch.tensor([kwargs["fwhm"]], dtype=torch.float32).to(self.device)
                else:
                    raise TypeError(f"Wrong FWHM format: {kwargs['fwhm']}, {type(kwargs['fwhm'])}")

            # Step 2: Use the superclass (i.e., original class) to compute the forward pass
            result_torch = super(VIPDecompositionWrapper, self).forward(cube_torch, angle_list=angle_list,
                                                                        verbose=verbose, **kwargs)

            return result_torch

        def forward_collapsed(self, cube: np.ndarray, angle_list: Any = None, verbose: bool = False, **kwargs: Any) -> np.ndarray:

            result_torch = self._forward(cube, angle_list=angle_list, verbose=verbose, **kwargs)

            # Step 2.5: VIP expects a 2D median combined image as the default output, ugh :(
            # TODO: this is not great for assessing the algo performance, but it is necessary for VIP
            result_torch = torch.mean(torch.median(result_torch,
                                                     dim=1, keepdim=False).values,
                                        dim=0, keepdim=False)  # torch does not support median over multiple dims

            # Step 3: Convert the result back to numpy and return
            result_numpy = result_torch.cpu().detach().numpy()

            return result_numpy

        def forward_full(self, cube: np.ndarray, angle_list: Any = None, verbose: bool = False, **kwargs: Any) -> np.ndarray:
            result_torch = self._forward(cube, angle_list=angle_list, verbose=verbose, **kwargs)

            result_numpy = result_torch.cpu().detach().numpy()

            return result_numpy


    # VIPDecompositionWrapper.__name__ = VIPDecompositionWrapper.__name__ + " - " + DecompositionClass.__name__
    VIPDecompositionWrapper.__name__ = DecompositionClass.__name__

    return VIPDecompositionWrapper


if __name__ == "__main__":
    pass
