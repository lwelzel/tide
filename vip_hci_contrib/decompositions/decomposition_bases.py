__author__ = "L. Welzel"
# __all__ = []

import torch
import tensorly as tl
import tensorly.decomposition as tld
import tntorch as tn
from abc import abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tl.set_backend("pytorch")


class AbstractBaseTensorDecomposition(torch.nn.Module):
    @abstractmethod
    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return tensor

    def _get_numel(self, factors):
        if isinstance(factors, (list, tuple)):
            return sum([f.numel() for f in factors])

class AbstractBaseMedianCODI(AbstractBaseTensorDecomposition):
    default_psf_settings = {

    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        median_psf = torch.median(tensor, dim=1, keepdim=True).values

        return median_psf


class AbstractBasePrincipalComponentAnalysis(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        "tol": 1e-5,
        'n_iter_pca_lr': 10,
        "ncomp": 500,
        "approx_method": "svd",
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # Reshape the tensor into a 2D matrix
        matrix = tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)

        # Get parameters from settings
        ncomp = kwargs.pop("ncomp")
        approx_method = kwargs.pop("approx_method")

        if approx_method == "pca_lr":
            n_iter_pca_lr = kwargs.pop("n_iter_pca_lr", 2)
            psf_model = self._psf_model_pca_lr(matrix, ncomp, n_iter_pca_lr=n_iter_pca_lr)
        elif approx_method == "svd":
            psf_model = self._psf_model_svd(matrix, ncomp)
        elif approx_method == "qr":
            psf_model = self._psf_model_qr(matrix, ncomp)
        else:
            raise NotImplementedError

        psf_model = psf_model.view(*tensor.shape)

        return psf_model

    def _psf_model_pca_lr(self, matrix: torch.Tensor, ncomp: int, n_iter_pca_lr: int, **kwargs) -> torch.Tensor:
        # Perform PCA using low-rank approximation
        u, s, v = torch.pca_lowrank(matrix, q=ncomp, niter=n_iter_pca_lr)

        # Project the matrix onto the space spanned by the first ncomp PCA components
        reduced_matrix = torch.matmul(matrix, v[:, :ncomp])

        # Reconstruct the data from the projected space
        psf_model = torch.matmul(reduced_matrix, v[:, :ncomp].t())

        return psf_model

    def _psf_model_svd(self, matrix: torch.Tensor, ncomp: int, **kwargs) -> torch.Tensor:
        # Perform SVD
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)

        # Keep only the first ncomp components
        # Note: In torch.linalg.svd, 'v' is returned as V^T, so we select rows, not columns
        reduced_matrix = torch.matmul(matrix, v[:ncomp, :].t())

        # Reconstruct the data
        # Again using the last ncomp rows of 'v', and since 'v' is V^T, we don't need to transpose it again
        psf_model = torch.matmul(reduced_matrix, v[:ncomp, :])

        return psf_model

    def _psf_model_qr(self, matrix: torch.Tensor, ncomp: int, **kwargs) -> torch.Tensor:
        # Perform QR decomposition on the matrix
        q, r = torch.linalg.qr(matrix)

        # Use only the first ncomp columns of Q and R for the low-rank approximation
        q_ncomp = q[:, :ncomp]
        r_ncomp = r[:ncomp, :]

        # Project the matrix onto the reduced space and then reconstruct it
        # Unlike PCA/SVD, QR doesn't inherently prioritize components by variance,
        # so this approach aims to reduce dimensions while keeping the reconstruction logic.
        psf_model = torch.matmul(q_ncomp, r_ncomp)

        return psf_model


class AbstractBaseRobustPCA(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        "tol": 1e-4,
        'n_iter_max': 100,
        # "ncomp_rotation": 5,
        # "ncomp_wavelength": 5,
        "ncomp": 5,
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        tensor = torch.moveaxis(tensor, 1, 0)
        ncomp = kwargs.pop("ncomp")
        low_rank, sparse_error = tld.robust_pca(tensor, **kwargs)
        low_rank[ncomp:] = 0.
        psf_model = torch.moveaxis(low_rank, 0, 1)
        return psf_model

    def _psf_model_split(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        tensor = torch.moveaxis(tensor, 1, 0)
        ncomp_rotation = kwargs.pop("ncomp_rotation")
        ncomp_wavelength = kwargs.pop("ncomp_wavelength")
        low_rank, sparse_error = tld.robust_pca(tensor, **kwargs)
        low_rank[ncomp_rotation:, ncomp_wavelength:] = 0.
        psf_model = torch.moveaxis(low_rank, 0, 1)
        return psf_model


class AbstractBaseCanonicalPolyadicDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': 5,
        'init': 'random',
        'n_iter_max': 100,
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _psf_model_tensorly(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs.get("sparsity", None) is not None:
            (weights, factors), sparse_component = tld.parafac(tensor, **kwargs)
            psf_model = tl.cp_to_tensor((weights, factors))
        else:
            shape = tensor.shape
            weights, factors = tld.parafac(
                tensor,#.view(shape[0], shape[1], -1),
                **kwargs)
            psf_model = tl.cp_to_tensor((weights, factors))#.view(*shape)

        return psf_model

    def _psf_model_tntorch(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        shape = tensor.shape
        t = tn.Tensor(
            tensor, #.view(shape[0], shape[1], -1),
            ranks_cp=kwargs["rank"]
        )
        psf_model = t.torch()#.view(*shape)
        return psf_model


class AbstractBaseCPDRobustTensorPowerIteration(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': 5,
        'n_repeat': 3,
        'n_iteration': 5,
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _psf_model_tensorly(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        weights, factors = tld.parafac_power_iteration(tensor, **kwargs)
        psf_model = tl.cp_to_tensor((weights, factors))
        return psf_model


class AbstractBaseTuckerDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [1, 1, 2, 2],
        'init': 'random',
        'n_iter_max': 100,
        "tol": 1e-3,
        # "mask_inner_rad_n_fwhm": 1.5,  # for injections its ~ 2
        # "mask": True,
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _psf_model_tensorly(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        __ = kwargs.pop("mask", None)
        mask = None
        __ = kwargs.pop("backend", None)
        shape = tensor.shape
        # print(f"shape: ({shape[0]}, {shape[1]}, {shape[-2] * shape[-1]})")
        (core, tucker_factors), rec_errors = tld.tucker(
            tensor.view(shape[0], shape[1], -1),
            mask=mask, return_errors=True, **kwargs)
        self.rec_errors.append(rec_errors)

        return tl.tucker_to_tensor((core, tucker_factors)).view(*shape)

    def _psf_model_tntorch(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        t = tn.Tensor(tensor, ranks_tucker=kwargs["rank"])
        return t.torch()


class AbstractBaseSVDTucker(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [2, 2, 5, 5],
        'cut_rank': [1, 1, 3, 3],
        'init': 'random',
        'n_iter_max': 100,
        "tol": 1e-4,
        "mask_inner_rad_n_fwhm": 1.5  # for injections its ~ 2
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _psf_model_tensorly(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        ncomp = kwargs.pop("cut_rank")

        if kwargs.get("mask"):
            __ = kwargs.pop("mask")
            mask = self.circular_mask(tensor, kwargs.pop("fwhm", self.fwhm), n=kwargs.pop("mask_inner_rad_n_fwhm", 1.5))
        else:
            __ = kwargs.pop("fwhm", None)
            __ = kwargs.pop("mask_inner_rad_n_fwhm", None)
            __ = kwargs.pop("mask", None)
            mask = None

        (core, tucker_factors), rec_errors = tld.tucker(tensor, mask=mask, return_errors=True, **kwargs)
        self.rec_errors.append(rec_errors)

        # truncate the Tucker decomposition according to T-HOSVD procedure
        lbda_cut, pa_cut, x_cut, y_cut = ncomp
        core = core[:lbda_cut, :pa_cut, :x_cut, :y_cut]
        tucker_factors = [factor[:, :cut_dim] for factor, cut_dim in zip(tucker_factors, ncomp)]

        return tl.tucker_to_tensor((core, tucker_factors))

    def _psf_model_tntorch(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        # t = tn.Tensor(tensor,
        #               ranks_tucker=kwargs["rank"]
        #               )
        # return t.torch()


class AbstractBaseNonNegativeTuckerDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [1, 1, 2, 2],
        'init': 'random',
        'n_iter_max': 100,
        "tol": 1e-4,
    }

    def _psf_model(self, tensor: torch.Tensor, which: str = "hals", **kwargs) -> torch.Tensor:
        """
        PSF model for non-negative Tucker decomposition.

        Args:
            tensor (torch.Tensor): Input tensor.
            which (str): PSF model type ("imu" or "hals").
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: PSF model tensor.

        """
        shape = tensor.shape

        if which == "imu":  # Iterative Multiplicative Update
            # Yong-Deok Kim and Seungjin Choi,
            # “Non-negative tucker decomposition”,
            # IEEE Conference on Computer Vision and Pattern Recognition s(CVPR), pp 1-8, 2007
            # TODO: it seems like this doesnt work very well for our data, why?
            core, tucker_factors = tld.non_negative_tucker(tensor, **kwargs)
        elif which == "hals":  # Hierarchical Alternating Least Squares
            core, tucker_factors = tld.non_negative_tucker_hals(
                tensor.view(shape[0], shape[1], -1),
                **kwargs
            )
        else:
            raise NotImplementedError
        tensor = tl.tucker_to_tensor((core, tucker_factors))
        return tensor.view(*shape)


class AbstractBaseTensorTrainDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [1, 1, 2, 2, 1],
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _psf_model_tensorly(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        shape = tensor.shape

        factors = tld.tensor_train(
            tensor.view(shape[0], shape[1], -1), #.view(-1, shape[-2], shape[-1]),  # .view(-1, shape[-2], shape[-1]),  # .view(shape[0], shape[1], -1),
            rank=kwargs["rank"]
        )
        tensor = tl.tt_to_tensor(factors).view(*shape)

        return tensor

    def _psf_model_tntorch(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        N = tensor.dim()
        if not hasattr(kwargs["rank"], '__len__'):
            kwargs["rank"] = [kwargs["rank"]] * (N - 1)
        else:
            kwargs["rank"] = kwargs["rank"][1:-1]
        assert len(kwargs["rank"]) == N - 1, f"Tucker ranks must be properly given. Got: {kwargs['rank']}"

        shape = tensor.shape
        t = tn.Tensor(tensor.view(shape[0], shape[1], -1),
                      ranks_tt=kwargs["rank"]
                      )
        tensor = t.torch().view(*shape)
        return tensor


class AbstractBaseTensorRingDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [5, 1, 20, 20, 5],
        # "permutation": [0, 1, 2, 3]  # original: lambda: 0, theta: 1, x: 2, y: 3
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        shape = tensor.shape
        factors = tld.tensor_ring(
            tensor, #.view(shape[0], shape[1], -1),
            **kwargs)
        tensor = tl.tr_to_tensor(factors).view(*shape)

        return tensor


class AbstractBaseRotationInvariantAnalysisOfVariancesDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [1, 1, 2, 2],
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # analysis of variances (ANOVA) decomposition
        # t = tn.Tensor(tensor, ranks_tucker=kwargs["rank"])
        shape = tensor.shape
        t = tn.Tensor(tensor.view(shape[0], shape[1], -1),
                      ranks_tt=kwargs["rank"])
        anova = tn.anova_decomposition(t)

        # cut anova
        # c, r, h, w = tn.symbols(4)
        c, r, p = tn.symbols(3)
        anova_cut = tn.mask(anova, ~tn.round(c & r))  # only keep terms that do not interact with rotation # tn.round(~h & ~w)

        # undo anova decomposition
        t_cut = tn.undo_anova_decomposition(anova_cut)
        psf_model = t_cut.torch().view(*shape)

        return psf_model

    def _companion_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # analysis of variances (ANOVA) decomposition
        t = tn.Tensor(tensor,
                      ranks_tt=[36, 200, 200]
                      )
        anova = tn.anova_decomposition(t)

        # cut anova
        c, r, h, w = tn.symbols(4)
        anova_cut = tn.mask(anova, tn.round(~c & ~r))  # only keep terms that do not interact with rotation & lbda

        # undo anova decomposition
        t_cut = tn.undo_anova_decomposition(anova_cut)
        companion_model = t_cut.torch()

        return companion_model


class AbstractBaseLowRankAnalysisOfVariancesDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [1, 1, 2, 2],
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # analysis of variances (ANOVA) decomposition
        # t = tn.Tensor(tensor, ranks_tucker=kwargs["rank"])
        t = tn.Tensor(tensor, ranks_tt=kwargs["rank"])
        anova = tn.anova_decomposition(t)

        # cut anova
        m = tn.weight_mask(4, range(3))  # mask that only includes n low-dimensional terms
        anova_cut = tn.mask(anova, m)  # only keep low rank terms

        # undo anova decomposition
        t_cut = tn.undo_anova_decomposition(anova_cut)
        psf_model = t_cut.torch()

        return psf_model


class AbstractBasePermutedTensorTrainDecomposition(AbstractBaseTensorDecomposition):
    default_psf_settings = {
        'rank': [1, 1, 2, 2, 1],
        "permutation": [0, 1, 2, 3]  # original: lambda: 0, theta: 1, x: 2, y: 3
    }

    def _psf_model(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _psf_model_tensorly(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        destination = kwargs.pop("permutation")
        tensor = torch.movedim(tensor, [0, 1, 2, 3], destination).contiguous()
        factors = tld.tensor_train(tensor, rank=kwargs["rank"])
        tensor = tl.tt_to_tensor(factors)
        tensor = torch.movedim(tensor, destination, [0, 1, 2, 3]).contiguous()
        return tensor

    def _psf_model_tntorch(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        N = tensor.dim()
        if not hasattr(kwargs["rank"], '__len__'):
            kwargs["rank"] = [kwargs["rank"]] * (N - 1)
        else:
            kwargs["rank"] = kwargs["rank"][1:-1]
        assert len(kwargs["rank"]) == N - 1, f"Tucker ranks must be properly given. Got: {kwargs['rank']}"

        destination = kwargs.pop("permutation")
        tensor = torch.movedim(tensor, [0, 1, 2, 3], destination).contiguous()
        t = tn.Tensor(tensor,
                      ranks_tt=kwargs["rank"]
                      )
        tensor = t.torch()
        tensor = torch.movedim(tensor, destination, [0, 1, 2, 3]).contiguous()
        return tensor


if __name__ == "__main__":
    pass
