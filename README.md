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

<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/e1b3d695-e877-4379-bbc1-21ade2359643" alt="Tensor Ring Decomposition" width="48%">
  <img src="https://github.com/lwelzel/tide/assets/29613344/5209c968-e0e8-4d4c-8711-5976cbee9f52" alt="STIM Map" width="48%">
</p>
<p align="center">
  <em>IFS observations of HR 8799 reduced with the Tensor Ring Decomposition, residuals (left) and STIM map (right).</em>
</p>

### Example

Decompositing the scaled observation tensor $\mathcal{X} \in \mathbb{R}^{I_\lambda \times I_\theta \times I_x \times I_y}$ using the Tensor Ring Decomposition, see the equations below, approximates the low-rank components of the coronagraphic PSF using four order-$3$ factors, $\mathcal{G}^{(i)}$, under the tensor trace operation. Subtracting the low-rank PSF model from the observations, rescaling and de-rotating results in the residual frame and STIM map shown in the figure above.

<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/f0cfabfb-6aa3-4bff-8503-c4f4912d8e9f" alt="TRD Eq. 1" width="75%">
</p>
<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/e0fba2e8-f494-4c62-a8b5-b8260115eb56" alt="TRD Eq. 2" width="30%">
</p>

Equivalently in tensor network notation:
<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/6fbb2636-fcd6-4591-821b-c61c847d0ce4" alt="TRD TND Eq. 3" width="48%">
</p>





