import torch
from typing import Type, Any, List, Optional, Tuple, Callable, NoReturn, Union
from tqdm import tqdm
from warnings import warn, showwarning
import matplotlib.pyplot as plt

from vip_hci_contrib.transforms.polar_transform import PolarTransformModule
from vip_hci_contrib.decompositions.polar_decomposition_codi_base import AbstractBasePolarTensorDecompositionCODI


class AbstractBaseAnnularTensorDecompositionCODI(AbstractBasePolarTensorDecompositionCODI):
    default_annulus_settings = {
        "inner_radius": None,
        "outer_radius": None,
        "annulus_width": 10,
        "n_overlap": 3,
        "single_annulus": False,
    }

    def __init__(self, cube: torch.Tensor, angle_list: torch.Tensor, scale_list: torch.Tensor, fwhm: torch.Tensor = None,
                 build_transform: bool = True, backend: str = "tensorly",
                 **kwargs):

        self.annulus_settings = {key: kwargs.pop(key, self.default_annulus_settings[key])
                                 for key in self.default_annulus_settings}

        __ = kwargs.pop("set_warp", None)
        for key in self.annulus_settings:
            __ = kwargs.pop(key, None)

        super(AbstractBaseAnnularTensorDecompositionCODI, self).__init__(cube=cube,
                                                                         angle_list=angle_list, scale_list=scale_list,
                                                                         fwhm=fwhm,
                                                                         backend=backend,
                                                                         **kwargs)
        if build_transform:
            # self.annulus_settings = {
            #     **self.default_annulus_settings,
            #     **{key: kwargs[key] for key in self.default_annulus_settings.keys() if key in kwargs}
            # }
            if self.annulus_settings["inner_radius"] is None:
                self.annulus_settings["inner_radius"] = 0.
            if self.annulus_settings["outer_radius"] is None:
                self.annulus_settings["outer_radius"] = self.polar_transform.radius_data_px * 0.9

            print(self.annulus_settings["inner_radius"], self.annulus_settings["outer_radius"])

            self.step_radii = self._get_mask_radii(**self.annulus_settings)
            self.polar_transforms = [
                PolarTransformModule(tensor=cube, inner_radius=inner, outer_radius=outer)
                for inner, outer in self.step_radii
            ]

            self.mask = torch.stack([
                pt.radius_mask
                for pt in self.polar_transforms
            ], dim=0)

        # self.residual = None
        # self.vmap_annular_psf_model = torch.vmap(
        #     self._annular_psf_model,
        #     in_dims=None,
        #     out_dims=None,
        #     chunk_size=1
        # )

        # DEFINE FORWARD FUNCTION
        # self.forward = self.med_comb_forward

        # TODO: transforms can only be called once forward() has been called.
        #  It is possible to implement behaviour to also add e.g. rotate, scale etc. to the hook,
        #  however it would require them to be defined as nn.Modules.

    def forward(self, cube: torch.Tensor, angle_list: Any = None, scale_list: Any = None,
                fwhm: Any = None, verbose: bool = False, psf: torch.Tensor = None, **kwargs) -> torch.Tensor:
        self.fwhm = fwhm

        # TODO: properly check if annulus settings are the same and recompute warps if they are not
        for key in self.annulus_settings:
            __ = kwargs.pop(key, None)
        __ = kwargs.pop("set_warp", None)

        print(cube.shape)

        # Prioritize settings passed during method call over stored ones
        method_settings = {**self.psf_settings, **kwargs}
        if self.annulus_settings["inner_radius"] is None:
            self.annulus_settings["inner_radius"] = self.fwhm.max()

        # CDI SCALING
        scaled_cube = self.scale(cube, scale_factors=self.scale_factors)
        scaled_cube = scaled_cube - torch.median(scaled_cube, keepdim=True, dim=1).values
        del cube

        # absolute genius idea: since we know the max number of estimates per pixel (n_overlap),
        # we can just make the residual tensor that high (dim 0) so that we can still make use of the median operation!
        residual = torch.zeros([*scaled_cube.shape])

        for i in tqdm(range(len(self.polar_transforms)), total=len(self.polar_transforms),
                      leave=False, desc="Modeling PSF annuli"):

            annulus_scaled_cube = self.polar_transforms[i].cart2pol(scaled_cube)

            # PSF model
            try:
                annulus_psf_model = self.psf_func(annulus_scaled_cube, **method_settings)
            except torch._C._LinAlgError as e:
                print(e)
                print("PSF model set to zero: annulus_psf_model = 0")
                annulus_psf_model = 0.

            # residuals
            annulus_residual = annulus_scaled_cube - annulus_psf_model

            annulus_residual = self.polar_transforms[i].pol2cart(annulus_residual)

            # here we use the property of the shifting annuli to preserve the full estimate
            residual += annulus_residual

        residual = residual / torch.clip(torch.sum(self.mask, dim=0, keepdim=False), min=1., max=None)
        # residual = residual * (torch.sum(self.mask, dim=0, keepdim=False) > 0.5)

        residual = self.rescale(residual, scale_factors=self.scale_factors)
        residual = self.derotate(residual, derot_angles=self.derot_angles)

        return residual

    def _annular_psf_model(self, polar_transform_idx: int, **method_settings) -> torch.Tensor:
        annulus_scaled_cube = self.polar_transforms[polar_transform_idx].cart2pol(self.scaled_cube)

        # PSF model
        try:
            annulus_psf_model = self.psf_func(annulus_scaled_cube, **method_settings)
        except torch._C._LinAlgError as e:
            # warn(e)
            # warn("PSF model set to zero: annulus_psf_model = torch.zeros_like(annulus_scaled_cube)")
            print(e)
            print("PSF model set to zero: annulus_psf_model = torch.zeros_like(annulus_scaled_cube)")
            annulus_psf_model = torch.zeros_like(annulus_scaled_cube)

        # residuals
        annulus_residual = annulus_scaled_cube - annulus_psf_model
        annulus_residual = self.polar_transforms[polar_transform_idx].pol2cart(annulus_residual)

        return annulus_residual

    @staticmethod
    def _get_mask_radii(inner_radius: float = 7., outer_radius: float = 100.,
                        annulus_width: int = 10, n_overlap: int = 2,
                        single_annulus=False, **kwargs) -> torch.Tensor:
        if single_annulus:
            return torch.tensor([[inner_radius, outer_radius]])

        # print(f"Radii: {inner_radius:.1f} -> {outer_radius:.1f}")

        n_steps = int((outer_radius - inner_radius) / (annulus_width / n_overlap)) + 1

        center_radii = torch.linspace(inner_radius, outer_radius, n_steps)[:-1]
        inner_radii = torch.clip(center_radii - (annulus_width / 2), min=0., max=None)
        outer_radii = center_radii + (annulus_width / 2)
        step_radii = torch.stack([inner_radii, outer_radii], dim=1)

        return step_radii


    def __med_comb_forward(self, cube: torch.Tensor, angle_list: Any = None, scale_list: Any = None,
                           fwhm: Any = None, verbose: bool = False, psf: torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This method is not refactored for the updated common polar transform and CODI.")
        self.fwhm = fwhm

        # Prioritize settings passed during method call over stored ones
        method_settings = {**self.psf_settings, **kwargs}

        __ = kwargs.pop("set_warp", None)
        if self.annulus_settings["inner_radius"] is None:
            self.annulus_settings["inner_radius"] = self.fwhm.max()

        # CDI SCALING
        scaled_cube = self.scale(cube, scale_factors=self.scale_factors)
        scaled_cube = scaled_cube - torch.median(scaled_cube, keepdim=True, dim=1).values
        scaled_cube = scaled_cube - torch.median(scaled_cube, keepdim=True, dim=0).values
        del cube

        self.mask = self.annulus_mask_2d(scaled_cube, **self.annulus_settings).view(-1, 1, 1,
                                                                                    *scaled_cube.shape[-2:])
        # TODO: this is stupid, remove the part that doesnt execute the first time around.
        #  Instead do it also with a pre-forward hook like the polar version.
        skip_because_first_time = False

        if self.warps is None:
            self.warps = [
                self._evaluate_warp(scaled_cube, self.derot_angles, self.lbda, initialRadius=inner, finalRadius=outer)
                for inner, outer in tqdm(self.step_radii.cpu().numpy(), total=len(self.step_radii),
                                         leave=False, desc="Creating warps")
            ]
            skip_because_first_time = True

        # absolute genius idea: since we know the max number of estimates per pixel (n_overlap),
        # we can just make the residual tensor that high (dim 0) so that we can still make use of the median operation!
        n_overlap = self.annulus_settings["n_overlap"]
        psf_model = torch.zeros([n_overlap, *scaled_cube.shape])

        if skip_because_first_time:
            return psf_model[0]

        for i, warp in tqdm(enumerate(self.warps), total=len(self.warps), leave=False, desc="Modeling PSF annuli"):
            annulus_scaled_cube = warp.to_polar(scaled_cube)

            # PSF model
            annulus_psf_model = self.psf_func(annulus_scaled_cube, **method_settings)
            psf_model[i % n_overlap] += warp.to_cart(annulus_psf_model) * self.mask[i]

        # TODO: previously we had to do this super shitty division to get the estimate,
        #  but now we can just use the median!
        #  This is because the estimates "wrap" around the dim 0 of the residual tensor.
        psf_model *= (
                torch.sum(self.mask, dim=0, keepdim=False).view(1, 1, 1, *scaled_cube.shape[-2:])
                > (n_overlap - 0.5)
        )
        psf_model = torch.median(psf_model, dim=0, keepdim=False).values

        # residuals
        residual = scaled_cube - psf_model

        residual = self._base_rescale(residual)
        residual = self._base_derotate(residual)

        return residual


class AbstractBaseSingleAnnulusTensorDecompositionCODI(AbstractBaseAnnularTensorDecompositionCODI):
    default_annulus_settings = {
        "annulus_width": 10.,
        "annulus_radius": 20.,
        "single_annulus": True,
    }

    def __init__(self, cube: torch.Tensor, angle_list: torch.Tensor, scale_list: torch.Tensor, fwhm: torch.Tensor = None,
                 backend: str = "tensorly",
                 **kwargs):

        self.annulus_settings = {key: kwargs.pop(key, self.default_annulus_settings[key])
                                 for key in self.default_annulus_settings}

        super(AbstractBaseAnnularTensorDecompositionCODI, self).__init__(
            cube=cube, angle_list=angle_list, scale_list=scale_list, fwhm=fwhm, backend=backend, build_transform=False,
            **kwargs
        )

        self.annulus_settings = {
            **self.default_annulus_settings,
            **{key: self.annulus_settings[key] for key in self.default_annulus_settings.keys() if key in self.annulus_settings},
        }

        inner_radius_unexpanded = self.annulus_settings["annulus_radius"] - self.annulus_settings["annulus_width"] / 2.
        inner_radius = inner_radius_unexpanded  # / self.scale_factors.max()
        self.annulus_settings["inner_radius"] = inner_radius

        outer_radius_unexpanded = self.annulus_settings["annulus_radius"] + self.annulus_settings["annulus_width"] / 2.
        outer_radius = outer_radius_unexpanded * self.scale_factors.max()
        self.annulus_settings["outer_radius"] = outer_radius

        assert self.annulus_settings["single_annulus"], "This class only supports single annulus decomposition."
        self.step_radii = self._get_mask_radii(**self.annulus_settings)

        self.polar_transform = PolarTransformModule(
            tensor=cube,
            inner_radius=self.annulus_settings["inner_radius"],
            outer_radius=self.annulus_settings["outer_radius"],
            upscale_factor=(2., 2.),
        )

        self.mask = self.polar_transform.radius_mask

        self.cartesian_mask = torch.logical_and(
            ~self.circular_mask_2d_custom_center(cube, 0, 0, inner_radius_unexpanded),
            self.circular_mask_2d_custom_center(cube, 0, 0, outer_radius_unexpanded).to(torch.bool)
        )

    def forward(self, cube: torch.Tensor, angle_list: Any = None, scale_list: Any = None,
                fwhm: Any = None, verbose: bool = False, psf: torch.Tensor = None, **kwargs) -> torch.Tensor:
        self.fwhm = fwhm

        # TODO: properly check if annulus settings are the same and recompute warps if they are not
        for key in self.annulus_settings:
            __ = kwargs.pop(key, None)
        __ = kwargs.pop("set_warp", None)

        # Prioritize settings passed during method call over stored ones
        method_settings = {**self.psf_settings, **kwargs}
        if self.annulus_settings["inner_radius"] is None:
            self.annulus_settings["inner_radius"] = self.fwhm.max()

        # CDI SCALING
        scaled_cube = self.scale(cube, scale_factors=self.scale_factors)
        # scaled_cube = scaled_cube - torch.median(scaled_cube, keepdim=True, dim=1).values

        del cube

        annulus_scaled_cube = self.polar_transform.cart2pol(scaled_cube)
        # print("annulus_scaled_cube shape:", annulus_scaled_cube.shape)

        # PSF model
        annulus_psf_model = self.psf_func(annulus_scaled_cube, **method_settings)

        # residuals
        annulus_residual = annulus_scaled_cube - annulus_psf_model

        residual = self.polar_transform.pol2cart(annulus_residual)

        residual = self.rescale(residual, scale_factors=self.scale_factors)
        residual = self.derotate(residual, derot_angles=self.derot_angles)

        return residual * self.cartesian_mask

    def forward_vals(self, *args, mask=torch.tensor([True]), **kwargs) -> torch.Tensor:
        residual = self.forward(*args, **kwargs)

        mask = torch.logical_and(mask, self.cartesian_mask)
        return residual[..., mask]

    def polar_forward(self, cube: torch.Tensor, angle_list: Any = None, scale_list: Any = None,
                fwhm: Any = None, verbose: bool = False, psf: torch.Tensor = None, **kwargs) -> torch.Tensor:
        self.fwhm = fwhm

        raise NotImplementedError

        # TODO: properly check if annulus settings are the same and recompute warps if they are not
        for key in self.annulus_settings:
            __ = kwargs.pop(key, None)
        __ = kwargs.pop("set_warp", None)

        # Prioritize settings passed during method call over stored ones
        method_settings = {**self.psf_settings, **kwargs}
        if self.annulus_settings["inner_radius"] is None:
            self.annulus_settings["inner_radius"] = self.fwhm.max()

        polar_annulus_cube = self.polar_transform.cart2pol(cube)
        # CODI SCALING
        scaled_polar_annulus_cube = self.polar_transform.scale(polar_annulus_cube, scale_factors=self.scale_factors)
        scaled_polar_annulus_cube = scaled_polar_annulus_cube - torch.median(scaled_polar_annulus_cube, keepdim=True, dim=1).values

        # PSF model
        polar_annulus_psf_model = self.psf_func(scaled_polar_annulus_cube, **method_settings)

        # residuals
        polar_annulus_residual = scaled_polar_annulus_cube - polar_annulus_psf_model

        polar_residual = self.polar_transform.rescale(polar_annulus_residual, scale_factors=self.scale_factors)

        polar_residual = self.polar_transform.derotate(polar_residual, derot_angles=self.derot_angles)

        residual = self.polar_transform.pol2cart(polar_residual)

        return residual * self.cartesian_mask


if __name__ == "__main__":
    pass
