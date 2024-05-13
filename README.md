# Direct Exoplanet Imaging with Tensor Decompositions.

<img align="left" width="33%" src="https://github.com/lwelzel/tide/assets/29613344/c6e89abe-5630-4648-9f64-35e928f222ff">

High-contrast imaging (HCI) is as essential pillar in the search for exoplanets. It relies heavily on advanced image 
processing techniques to differentiate planetary flux from the coronagraphic point spread function (PSF) of the host 
star and nuisance components. Angular-spectral differential imaging (ASDI) is a key method to induce diversity in HCI 
observations by utilizing integral field spectrographs. Traditional methods like Principal Component Analysis (PCA) 
have been instrumental but are limited by their inability to fully capture the complex, multi-modal relationships of 
the diversity exploited by ASDI data processing.

This study introduces tensor decomposition methods as robust alternatives to PCA for direct imaging of 
exoplanets, aiming to preserve and leverage the multi-modal structure of the observational data more effectively.

This project interprets the ASDI data as high-order tensors with cross-couplings in the spectral, 
temporal and spatial modes which are disrupted by flattening the tensors into a matrix for factorization. Instead of 
using PCA, the observation tensors are decomposed into low-rank factors using tensor decompositions, including the 
Canonical-Polyadic, Tucker, Tensor Train, and Tensor Ring Decomposition. These decompositions compute a low-rank PSF 
model while preserving the structure and higher-order relationships of the modes. The new methods are assessed on 
synthetic and real observations from the SPHERE instrument on the Very Large Telescope.

The tensor-based methods demonstrated a capacity to maintain the integrity of the ASDI data's
multi-modal structure, and capture cross-modal interactions. They provide a more adaptable framework for PSF 
modeling and subtraction. Evaluation against traditional PCA show comparable performance. Tensor decomposition 
methods increase the flexibility as well as interpretability of factorizations and extend existing methods in direct 
exoplanet imaging. These findings suggest that tensor decompositions can further advance HCI post-processing and 
make deep learning on HCI datasets tenable.
