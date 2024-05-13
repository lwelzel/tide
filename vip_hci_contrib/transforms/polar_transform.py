import torch
import kornia
import polarTransform
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from vip_hci.var import frame_center
from typing import Tuple, Union
from kornia.geometry.conversions import cart2pol, pol2cart
from kornia.geometry.transform import remap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device('cuda')


class BasePolarTransformModule(torch.nn.Module):
    def __init__(self, tensor: torch.Tensor,
                 inner_radius: Union[float, int] = 0.1, outer_radius: Union[float, int, None] = None,
                 upscale_factor: Tuple[float, float] = (1., 2.)):
        super(BasePolarTransformModule, self).__init__()

        self.upscale_factor = upscale_factor

        # inner radius, in px or px fraction
        self.inner_radius = inner_radius

        # get max radius up to which there is non-zero data
        self.radius_data, self.radius_data_deprojected = self.compute_data_radius(tensor)
        self.radius_data_px = int(self.radius_data * np.ceil(tensor.shape[-2] / 2.))
        safety_margin = 1  # in px

        cut_radius = self.radius_data_deprojected
        if outer_radius is not None:
            outer_radius_fraction = outer_radius / np.floor(tensor.shape[-2] / 2.)
            if isinstance(outer_radius, float):
                cut_radius = np.minimum(self.radius_data_deprojected, outer_radius_fraction)
            elif isinstance(outer_radius, torch.Tensor):
                cut_radius = np.minimum(self.radius_data_deprojected, outer_radius_fraction.item())
        self.outer_radius_gap_px = int(
            np.clip(((1. - cut_radius)
                     * np.ceil(tensor.shape[-2] / 2.)) - safety_margin,
                    0, np.inf
                    )
        )

        tensor = self._cut_tensor(tensor)
        self.radius_data = np.sqrt(2.)

        # outer radius, in px or px fraction
        if outer_radius is None:
            outer_radius = self.radius_data_px
        self.outer_radius = outer_radius

        # SHAPES
        # shape: (*, x, y)
        self.original_shape = tensor.shape

        # define radii
        self.radius_normal = 1.
        self.radius_diagonal = np.sqrt(2.)

        self.px_normal = int(np.ceil(tensor.shape[-2] / 2.))
        self.radius_normal_px = int(self.radius_normal * self.px_normal)
        self.radius_diagonal_px = int(self.radius_diagonal * self.px_normal)

        # shape: (*, x, y)
        self.cartesian_shape = torch.tensor([*tensor.shape])
        self.cartesian_center = torch.floor(torch.tensor([
            tensor.shape[-2] / 2.,
            tensor.shape[-1] / 2.
        ])).to(torch.int)

        # polar shape: (*, rho, phi)
        cross = torch.tensor([
            [self.cartesian_shape[-1] - 1, self.cartesian_center[1]],
            [0, self.cartesian_center[1]],
            [self.cartesian_center[0], self.cartesian_shape[-2] - 1],
            [self.cartesian_center[0], 0]
        ])

        self.radius_size = torch.ceil(self.upscale_factor[-2] * torch.abs(
            cross - self.cartesian_center
        ).max() * 2. * (self.outer_radius - self.inner_radius) / self.radius_diagonal_px).to(torch.int64).item()

        self.angle_size = int(self.upscale_factor[-1] * torch.max(self.cartesian_shape[-2:]))

        # shape: (*, rho, phi)
        self.polar_shape = torch.tensor([
            *tensor.shape[:-2],
            self.radius_size,
            self.angle_size,
        ]).to(torch.int)

        # PIXEL FLOW MAPS
        (self.map_cartesian_to_polar_x, self.map_cartesian_to_polar_y), self.phase_mask = self.build_cart2pol_maps()
        (self.map_polar_to_cartesian_x, self.map_polar_to_cartesian_y), self.radius_mask = self.build_pol2cart_maps()

        self.radius_mask = self._expand_tensor(self.radius_mask)

        # self.show_maps()

        self._unsqueeze_maps()




    def cart2pol(self, cart_tensor: torch.Tensor, cut=True) -> torch.Tensor:

        if cut:
            cart_tensor = self._cut_tensor(cart_tensor)

        polar_tensor = remap(
            cart_tensor.view(1, -1, *cart_tensor.shape[-2:]),
            map_x=self.map_cartesian_to_polar_x, map_y=self.map_cartesian_to_polar_y,
            align_corners=True,
            normalized_coordinates=False,
            padding_mode="border",
            mode="bilinear",
        )
        polar_tensor = polar_tensor.view(*cart_tensor.shape[:-2], *polar_tensor.shape[-2:])
        return (polar_tensor * self.phase_mask).contiguous()

    def pol2cart(self, polar_tensor: torch.Tensor, cut=True) -> torch.Tensor:
        # kornia expects the incoming tensor to be in (*, phi, rho) to be aligned with the x and y map,
        # but the tensor comes in as (*, rho, phi) with our convention, so we switch the map dimensions
        cart_tensor = remap(
            polar_tensor.view(1, -1, *polar_tensor.shape[-2:]),
            map_x=self.map_polar_to_cartesian_y, map_y=self.map_polar_to_cartesian_x,
            align_corners=True,
            normalized_coordinates=False,
            padding_mode="border",
            mode="bilinear",
        )

        cart_tensor = cart_tensor.view(*polar_tensor.shape[:-2], *cart_tensor.shape[-2:])

        if cut:
            cart_tensor = self._expand_tensor(cart_tensor)

        return (cart_tensor * self.radius_mask).contiguous()

    def pol2cart_masked_values(self, polar_tensor: torch.Tensor) -> torch.Tensor:
        cart_tensor = self.pol2cart(polar_tensor=polar_tensor, cut=False)
        return cart_tensor[..., self.radius_mask]

    def car2pol_masked_values(self, cart_tensor: torch.Tensor) -> torch.Tensor:
        polar_tensor = self.cart2pol(cart_tensor=cart_tensor, cut=False)
        return polar_tensor[..., self.phase_mask]

    def build_cart2pol_maps(self):
        rho = self.linspace(self.inner_radius, self.outer_radius, self.radius_size)
        phi = self.linspace(0., 2. * torch.pi, self.angle_size)

        rr, pp = torch.meshgrid(rho, phi, indexing="ij")

        map_cartesian_to_polar_x = rr * torch.cos(pp) + self.cartesian_center[0]
        map_cartesian_to_polar_y = rr * torch.sin(pp) + self.cartesian_center[1]

        r_max = torch.sqrt(1. + 2 *
                           torch.minimum(torch.square(torch.sin(phi)),
                                         torch.square(torch.cos(phi)),
                                         )
                           ) * self.radius_normal_px

        phase_mask = torch.lt(rr, r_max)

        return (map_cartesian_to_polar_x, map_cartesian_to_polar_y), phase_mask

    def build_pol2cart_maps(self):
        scale_radius = self.polar_shape[-2] / (self.outer_radius - self.inner_radius)
        scale_angle = self.polar_shape[-1] / (2. * torch.pi)

        x = self.linspace(- self.cartesian_shape[-2] / 2., self.cartesian_shape[-2] / 2., self.cartesian_shape[-2])
        y = self.linspace(- self.cartesian_shape[-1] / 2., self.cartesian_shape[-1] / 2., self.cartesian_shape[-1])

        yy, xx = torch.meshgrid(x, y, indexing="ij")

        # get x and y pixel flow from polar to cartesian
        map_polar_to_cartesian_x = torch.sqrt(torch.square(xx) + torch.square(yy))
        radius_mask = torch.logical_and(
            torch.gt(map_polar_to_cartesian_x, self.inner_radius),
            torch.lt(map_polar_to_cartesian_x, self.outer_radius)
        )
        map_polar_to_cartesian_x = map_polar_to_cartesian_x - self.inner_radius
        map_polar_to_cartesian_x = map_polar_to_cartesian_x * scale_radius

        map_polar_to_cartesian_y = torch.atan2(yy, xx)
        map_polar_to_cartesian_y = (map_polar_to_cartesian_y + 2. * torch.pi) % (2. * torch.pi)
        map_polar_to_cartesian_y = map_polar_to_cartesian_y * scale_angle

        return (map_polar_to_cartesian_x, map_polar_to_cartesian_y), radius_mask

    def _unsqueeze_maps(self):
        # unsqueeze for kornia
        self.map_cartesian_to_polar_x = self.map_cartesian_to_polar_x.unsqueeze(0).contiguous()
        self.map_cartesian_to_polar_y = self.map_cartesian_to_polar_y.unsqueeze(0).contiguous()

        self.map_polar_to_cartesian_x = self.map_polar_to_cartesian_x.unsqueeze(0).contiguous()
        self.map_polar_to_cartesian_y = self.map_polar_to_cartesian_y.unsqueeze(0).contiguous()

    @staticmethod
    def linspace(start, end, steps):
        return torch.linspace(start, end, steps + 1)[:-1].contiguous()

    def _cut_tensor(self, tensor: torch.Tensor, gap: int=None) -> torch.Tensor:
        # tensor must be cartesian
        if gap is None:
            gap = self.outer_radius_gap_px

        if gap == 0:
            return tensor
        cut_tensor = tensor[..., gap:-gap, gap:-gap]

        return cut_tensor.contiguous()

    def _expand_tensor(self, tensor: torch.Tensor, gap: int=None) -> torch.Tensor:
        # tensor must be cartesian
        if gap is None:
            gap = self.outer_radius_gap_px

        expanded_tensor = torch.nn.functional.pad(tensor, (gap, gap, gap, gap), mode="constant", value=0)

        return expanded_tensor.contiguous()

    @staticmethod
    def compute_data_radius(tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Compute the radius up which to there is meaningful data in the image cube.
        The radius is computed as the maximum radius up to which there is non-zero data in the cube.
        Then the position of the element associated with the maximum radius is used to compute the deprojected radius.
        The deprojected radius can be used to trim the cube.
        """
        try:
            # data radius is in units of normal radii
            x = torch.linspace(-1., 1., tensor.shape[-2])
            y = torch.linspace(-1., 1., tensor.shape[-1])

            yy, xx = torch.meshgrid(x, y, indexing="ij")
            radius = torch.sqrt(torch.square(xx) + torch.square(yy))
            mask = torch.eq(torch.sum(torch.abs(tensor), dim=(0, 1)), 0.)
            masked_radius = ~mask * radius  # invert mask: elements=0: True -> elements!=0: True

            radius_data = masked_radius.max().item()

            max_radius_idx_flat = torch.argmax(masked_radius)  # freaking torch and flat argmax?!

            max_radius_idx = [(max_radius_idx_flat % tensor.shape[-2]),
                              max_radius_idx_flat // tensor.shape[-2]]
            theta_max = torch.atan2(max_radius_idx[1] - tensor.shape[-2] / 2,
                                    max_radius_idx[0] - tensor.shape[-1] / 2)

            data_radius_deprojected = torch.maximum(
                torch.abs(torch.cos(theta_max) * radius_data),
                torch.abs(torch.sin(theta_max) * radius_data),
            ).item()
        except RuntimeError:
            radius_data = np.sqrt(2.)
            data_radius_deprojected = 1.

        return radius_data, data_radius_deprojected

    def show_maps(self, rr=None, pp=None, xx=None, yy=None):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(24, 8), constrained_layout=True)

        if rr is not None:
            ax = axes[0, 0]
            im = ax.imshow(rr.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
            plt.colorbar(im, ax=ax)
            ax.set_title("rr")

        if pp is not None:
            ax = axes[1, 0]
            im = ax.imshow(pp.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
            plt.colorbar(im, ax=ax)
            ax.set_title("pp")

        ax = axes[0, 1]
        im = ax.imshow(self.map_cartesian_to_polar_x.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("map_cartesian_to_polar_x")

        ax = axes[1, 1]
        im = ax.imshow(self.map_cartesian_to_polar_y.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("map_cartesian_to_polar_y")

        if xx is not None:
            ax = axes[0, 2]
            im = ax.imshow(xx.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
            plt.colorbar(im, ax=ax)
            ax.set_title("xx")

        if yy is not None:
            ax = axes[1, 2]
            im = ax.imshow(yy.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
            plt.colorbar(im, ax=ax)
            ax.set_title("yy")

        ax = axes[0, 3]
        im = ax.imshow(self.map_polar_to_cartesian_x.squeeze().cpu().numpy(), origin="lower", cmap="seismic",
                       # vmin=-1, vmax=1.
                       )
        plt.colorbar(im, ax=ax)
        ax.set_title("map_polar_to_cartesian_x")

        ax = axes[1, 3]
        im = ax.imshow(self.map_polar_to_cartesian_y.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("map_polar_to_cartesian_y")

        plt.show()


class PolarTransformModule(BasePolarTransformModule):
    @staticmethod
    def scale(tensor: torch.Tensor, scale_factors: torch.Tensor) -> torch.Tensor:
        scale_factors[0, :] = 1.
        scaled_cube = kornia.geometry.transform.scale(
            tensor,
            scale_factor=scale_factors,
            mode="bilinear"
        )
        return scaled_cube

    @staticmethod
    def rescale(tensor: torch.Tensor, scale_factors: torch.Tensor) -> torch.Tensor:
        scale_factors[0, :] = 1.
        rescaled_cube = kornia.geometry.transform.scale(
            tensor=tensor,
            scale_factor=1. / scale_factors,
            mode="bilinear"
        )
        return rescaled_cube

    @staticmethod
    def rotate(tensor: torch.Tensor, derot_angles: torch.Tensor) -> torch.Tensor:
        temp_tensor = torch.zeros_like(tensor)

        angle_scale = tensor.shape[-2] / 360.
        derotate_angles_px = derot_angles * angle_scale
        max_derot_angle = torch.ceil(torch.max(derotate_angles_px)).to(torch.int64)

        temp_tensor[..., :-max_derot_angle] = temp_tensor[..., max_derot_angle:]
        temp_tensor[..., :max_derot_angle] = tensor[..., -max_derot_angle:]

        pad = 3
        temp_tensor = torch.functional.F.pad(temp_tensor, (pad, pad, 0, 0), mode="constant", value=0)
        temp_tensor[..., :pad] = temp_tensor[..., -2 * pad:-pad]
        temp_tensor[..., -pad:] = temp_tensor[..., pad:2 * pad]

        translation = torch.zeros(derotate_angles_px.shape[0], 2)
        translation[:, 0] = derotate_angles_px - max_derot_angle

        derot_cube = kornia.geometry.transform.translate(
            torch.moveaxis(temp_tensor, 1, 0),
            translation=translation,
            mode="bilinear",

        )
        derot_cube = torch.moveaxis(derot_cube, 0, 1)

        derot_cube = derot_cube[..., pad:-pad]

        return derot_cube.contiguous()

    def derotate(self, tensor: torch.Tensor, derot_angles: torch.Tensor) -> torch.Tensor:
        rot_cube = self.rotate(tensor, derot_angles=-derot_angles)
        return rot_cube


def plot_cords(img, title="", vmin=None, vmax=None):
    if img.ndim == 4:
        img = torch.mean(img.clone().detach(), dim=(0, 1))
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)

    im = ax.imshow(img.cpu().numpy(), origin="lower",
                   cmap="seismic",
                   vmin=vmin, vmax=vmax
                   )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)

    plt.show()


def polar_transform_test():
    from pathlib import Path
    from helpers.experiment_runner import FileHandler
    from kornia.geometry.conversions import cart2pol, pol2cart
    from kornia.geometry.transform import remap
    d = f"HIP83547_OBS_YJ_2015-07-01_ifs_convert_dc2_PUBLIC_199231"
    data_dir = Path(f"/home/lwelzel/Documents/git/tensor_asdi/data/SPHERE_DC_DATA/{d}/")
    results_dir = Path("/home/lwelzel/Documents/git/tensor_asdi/data/results")

    file_handler = FileHandler(data_dir, results_dir=results_dir)

    cube = torch.tensor(file_handler.cube)
    cube[torch.isnan(cube)] = 0.
    # cube = torch.ones_like(cube)

    x = torch.linspace(-1., 1., 290 + 1)[:-1]
    y = torch.linspace(-1., 1., 290 + 1)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    map_x, map_y = cart2pol(xx, yy)

    # cube[:, :] = map_x
    # cube = cube # + torch.randn_like(cube) * 5e-1

    # c = 50
    # cube = cube[:, :, c:-c, c:-c].clone().detach()

    plot_cords(cube, "original")

    print(cube.shape)

    polar_transform_module = PolarTransformModule(cube)
    polar_image = polar_transform_module.cart2pol(cube)
    polar_transform_module.show_polar_image(polar_image, "Polar Image")
    cart_image = polar_transform_module.pol2cart(polar_image)
    print(cart_image.shape)
    plot_cords(cart_image, "reconstructed")
    diff = cube - cart_image

    plot_cords(diff, "error",
               vmax=torch.quantile(torch.mean(diff, dim=(0, 1)), q=1.-1.e-3),
               vmin=torch.quantile(torch.mean(diff, dim=(0, 1)), q=1.e-3)
               )



if __name__ == "__main__":
    polar_transform_test()
