# Direct Exoplanet Imaging with Tensor Decompositions.

<img align="left" width="33%" src="https://github.com/lwelzel/tide/assets/29613344/c6e89abe-5630-4648-9f64-35e928f222ff" alt="post-processed HR 8799 observations (IRDIS) with overlaid tensor ring diagram">

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
  <em><strong>IFS observations of HR 8799 reduced with the Tensor Ring Decomposition, residuals (left) and STIM map (right).</strong></em>
</p>

### Angular-Spectral Differential Imaging

Combined Differential Imaging (CODI) is an observational technique that exploits angular and spectral diversity to differentiate the stellar PSF from the companion. By using both angular and spectral diversity the PSF of the host star is modeled by scaling the observations so that the stellar PSF is aligned along the angular and spectral dimension but the companion is misaligned along both dimensions. This process is shown in the figure below. The stellar PSF is then modeled as the quasi-static part of the observations. The PSF model is then subtracted from the observations, ideally leaving only the companion signal.

<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/c36f7581-d229-45f1-af7d-455e2ae0f17d" alt="original/rescaled ASDI observations" width="48%">
  <img src="https://github.com/lwelzel/tide/assets/29613344/1424c415-f640-4bfb-b36b-6ffa4fed371e" alt="scaled ASDI observations" width="48%">
</p>
<p align="center">
  <em><strong>Schematic observations obtained by angular-spectral differential imaging.</strong> Each frame (gray border) in the 3-by-3 grids is an image of the observed solar system at wavelength &lambda; and parallactic angle &theta;. The coordinate system in each frame is not wavelength and angle, but instead (projected) distances e.g. right ascension and declination. The PSF   and speckle pattern are shown in red, and an off-axis source (like an exoplanet) is shown in blue, and its trajectory through the observation cube is shown as a dashed blue arc. The center of each frame, coinciding with the position of the star, is indicated by a black circle. While only 9 frames are shown here, full observations typically consist of thousands of frames. <strong>Left: pre-processed observations.</strong> The PSF and speckle pattern spread out with increasing wavelength due to the diffraction of light. Due to the rotation of the earth under the sky, the off-axis source moves on an arc through the observations. <strong>Right: scaled observations.</strong> By scaling the frames by the ratio of a reference wavelength &lambda;<sub>0</sub> (typically the largest wavelength in an observation) over the wavelength of a frame, also called the scale factor &lambda;<sub>0</sub> / &lambda; = s, the PSF and speckle pattern is aligned throughout the entire cube. This also misaligns off-axis sources, both radially and azimuthally. Ideally, the PSF and speckle pattern is now the same in every frame and can be easily modeled.</em>

Typically, the PSF model is found using matrix Principal Component Analysis (PCA) which relies on the truncated matrix Singular Value Decomposition (SVD). The figure below illustrates this method on the simpler case of spectral differntial imaging.

<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/2ec07fb0-4283-474f-ba4e-547e2e057ae3" alt="SDI with PCA" width="100%">
</p>
<p align="center">
  <em>Post-processing spectral differential imaging observations with matrix principal component analysis (PCA) using the truncated matrix singular value decomposition (SVD).</em>
</p>

First the observations are prepared by aligning the coronagraphic point spread function in the observations ![X_{Obs}](https://latex.codecogs.com/svg.latex?\mathcal{X}_{Obs}). This misaligns the planet (P). In this schematic, the observations of the star and planet are frames with $3 \times 3$ pixels and cover $5$ spectral bands from the optical (blue) to the infrared (red). The observations are thus a $5 \times 3 \times 3$ data cube. The observations are then flattened into a $9\times 5$ matrix. This is necessary to use matrix-PCA to find a PSF model. Flattening the observations disrupts the relationships between the data points. The flattened observation matrix ![X](https://latex.codecogs.com/svg.latex?\mathbf{X}) is factorized using the SVD. This factorization is exact so that ![X=U \Sigma V^{\top}](https://latex.codecogs.com/svg.latex?\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}) holds. The columns of ![U](https://latex.codecogs.com/svg.latex?\mathbf{U}) are the principal components (PC) and the rows of ![V^{\top}](https://latex.codecogs.com/svg.latex?\mathbf{V}^{\top}) are the principal directions, ordered from most "important" (dark) to least "important" (light). The "importance" of each PC is given by its singular values on the diagonal of ![\Sigma](https://latex.codecogs.com/svg.latex?\boldsymbol{\Sigma}). The unimportant PC contain typically only noise and the planet. The white cells are zero. To model the PSF without including the planet, only the important PC are considered, thus truncating the matrices. In this example 4 of the 5 PC are included in the model. This is equivalent to finding a rank-4 model ![M_{PSF}](https://latex.codecogs.com/svg.latex?\mathbf{M}_{PSF}) that optimally approximates the rank-5 observations ![X_{Obs}](https://latex.codecogs.com/svg.latex?\mathbf{X}_{Obs}) in a least-squares sense. The matrix PSF model ![M_{PSF}](https://latex.codecogs.com/svg.latex?\mathbf{M}_{PSF}) can then be tensorized into its original data-cube shape. By including only a few PC in the model, the planet has been successfully removed from the data. Subtracting the PSF model from the original observations leaves only the flux from the planet and noise.

Because reshaping the observations into a matrix disrupts the relationships between the modes we propose to model the observations as higher order tensors using tensor decompositions. Tensor decompositions generalize PCA and SVD to higher-order data. Below the idea behind this method using the canonical-polyadic decomposition (CPD) is illustrated on the example of an observation cube optained by spectral differential imaging (only spectral diversity).

<p align="center">
  <img src="https://github.com/lwelzel/tide/assets/29613344/9e96279e-82c7-4e97-91f0-57e6d1f8939e" alt="SDI with CPD" width="100%">
</p>
<p align="center">
  <em>Post-processing spectral differential imaging observations with tensor methods using the Canonical-Polyadic decomposition (CPD).</em>
</p>

The observations are pre-processed as before. However, instead of modeling the PSF as a matrix, the tensor methods find a model that has the same shape as the original data. In the case of the CPD, this model is the sum of vector products. Each vector product has a factor for the spectrum ![u^{(\lambda)}_{n}](https://latex.codecogs.com/svg.latex?\mathbf{u}^{(\lambda)}_{n}), right ascension ![u^{(x)}_{n}](https://latex.codecogs.com/svg.latex?\mathbf{u}^{(x)}_{n}) and declination separation ![u^{(y)}_{n}](https://latex.codecogs.com/svg.latex?\mathbf{u}^{(y)}_{n}). These vector products are conceptually similar to the principal components of the singular value decomposition. However, because the CPD "keeps track" of the original data shape, it is able to distinguish between features in the spectral and spatial modes.


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





