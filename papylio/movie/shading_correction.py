#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shading correction utilities including the BaSiC optimizer.

This module implements the BaSiC algorithm and supporting functions for
estimating flatfield (illumination) and darkfield corrections for microscopy
image stacks.
"""

__all__ = ['BaSiC']

from pathlib import Path
import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
import tqdm
from numba import njit

# From:
# linum-uqam/PyBaSiC: v1.0.0
# Joël Lefebvre
# 10.5281/zenodo.7305570
# https://github.com/linum-uqam/PyBaSiC

class BaSiC(object):
    """BaSiC (Background and Shading Intensity Correction) optimizer.

    Implements the BaSiC algorithm for correcting spatially-varying illumination
    and background intensity in microscopy images. This is an improved version
    of the original MATLAB implementation, using iterative reweighted L1 minimization.

    References
    ----------
    linum-uqam/PyBaSiC: v1.0.0
    Joël Lefebvre
    10.5281/zenodo.7305570
    https://github.com/linum-uqam/PyBaSiC

    Notes
    -----
    - Estimates flatfield (illumination correction) and optionally darkfield (offset)
    - Uses DCT (Discrete Cosine Transform) for regularization
    - Employs iterative reweighting for robust L1 minimization
    """
    def __init__(self, input, estimate_darkfield=False, extension=".tif", verbose=False,
                 working_size=128, epsilon=0.1, l_s=None, l_d=None, reweighting_tolerance=1e-3,
                 max_reweightingIterations=10):
        """Initialize BaSiC optimizer.

        Parameters
        ----------
        input : str, Path, list, or np.ndarray
            Input data source. Can be:
            - Path (str or Path) to directory containing images
            - List of image file paths
            - List of numpy arrays (images)
            - Numpy array stack of shape (N_images, height, width)
        estimate_darkfield : bool, optional
            If True, estimate darkfield (offset). If False, assume offset=0 (default: False)
        extension : str, optional
            File extension to search for when input is a directory (default: ".tif")
        verbose : bool, optional
            If True, print progress information (default: False)
        working_size : int, optional
            Image downsampling size for faster computation. Images are resampled to
            (working_size, working_size) (default: 128)
        epsilon : float, optional
            Stability parameter for reweighted L1-norm (default: 0.1)
        l_s : float, optional
            Regularization parameter for flatfield. If None, auto-computed (default: None)
        l_d : float, optional
            Regularization parameter for darkfield. If None, auto-computed (default: None)
        reweighting_tolerance : float, optional
            Convergence tolerance for L1 reweighting iterations (default: 1e-3)
        max_reweightingIterations : int, optional
            Maximum number of reweighting iterations (default: 10)

        Raises
        ------
        ValueError
            If input format is not recognized
        AssertionError
            If no files found when input is a directory
        """
        self.input_type = None
        self.extension = extension
        if isinstance(input, str) or isinstance(input, Path): # Directory
            self.directory = input
            self._sniff_input() # Get a list of files
            self.input_type = "directory"
        elif isinstance(input, np.ndarray):
            self.img_stack = input
            self.input_type = "images_stack"
        elif isinstance(input, list) and (isinstance(input[0], str) or isinstance(input[0], Path)):
            self.files = input
            self.input_type = "files_list"
        elif isinstance(input, list) and isinstance(input[0], np.ndarray):
            self.img_stack = np.array(input)
            self.input_type = "images_list"
        else:
            raise "input should either be a directory, a list of ndarrays, or a ndarray stack."

        # Optimizer parameters
        self.working_size = working_size  # px : image resampling size to accelerate learning.
        self.epsilon = epsilon  # Iterative reweighted L1-norm stability parameter
        self.l_s = l_s  # flat-field regularization parameter (set automatically if None)
        self.l_d = l_d  # dark-field regularization parameter (set automatically if None)
        self.reweighting_tolerance = reweighting_tolerance
        self.max_reweightingIterations = max_reweightingIterations
        self.reweighting_iteration = 0
        self.estimate_darkfield = estimate_darkfield
        self.verbose = verbose

    def _sniff_input(self):
        """Discover and sort image files in the input directory.

        Searches for files with the specified extension and sorts them.
        Called automatically when input is a directory.
        """
        # Get a list of tiles to process
        directory = Path(self.directory).resolve()
        file_list = list(directory.glob(f"*{self.extension}"))
        file_list.sort()
        self.files = file_list
        assert len(self.files) > 0, "No files were found in the input directory. Make sure you provided the right path and file extension. Aborting."

    def _load_images(self, img_stack = None):
        """Load and preprocess image stack.

        Loads images from files (if needed) and resamples to working_size
        for faster computation.

        Parameters
        ----------
        img_stack : np.ndarray, optional
            Pre-loaded image stack. If None, loads from files (default: None)
        """
        # Load the stack
        if img_stack is None:
            img_stack = []
            if self.verbose:
                gen = tqdm.tqdm(self.files, "Loading the images")
            else:
                gen = self.files
            for this_file in gen:
                img = cv2.imread(str(this_file), cv2.IMREAD_ANYDEPTH)
                img_stack.append(img)
            self.img_stack = np.array(img_stack)
        else:
            self.img_stack = img_stack
        self.n_images = self.img_stack.shape[0]

        # Resample the images to accelerate learning
        self.image_shape = self.img_stack.shape[1::]
        new_shape = tuple([self.working_size]*2)
        img_stack_p = np.zeros([self.n_images, *new_shape], dtype=self.img_stack.dtype)
        if self.working_size > self.image_shape[0]:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        for i in range(self.n_images):
            img = self.img_stack[i, ...].squeeze()
            img_stack_p[i, ...] = cv2.resize(img.T, new_shape, interpolation=interpolation).T
        self.img_stack_resized = img_stack_p.astype(np.float32)

    def normalize(self, img, clip=True, epsilon=1e-6):
        """Apply shading correction to an image.

        Normalizes an image using the estimated flatfield and darkfield.

        Parameters
        ----------
        img : np.ndarray
            Input image to normalize
        clip : bool, optional
            If True, clip output to valid range of input dtype (default: True)
        epsilon : float, optional
            Small value for numerical stability (default: 1e-6)

        Returns
        -------
        np.ndarray
            Corrected image in the same dtype as input
        """
        img_p = (img.astype(np.float32) - self.darkfield_fullsize) / (self.flatfield_fullsize + epsilon)
        if clip and not(img.dtype in [np.float32, np.float64]):

            img_p[img_p < np.iinfo(img.dtype).min] = np.iinfo(img.dtype).min
            img_p[img_p > np.iinfo(img.dtype).max] = np.iinfo(img.dtype).max

        return img_p.astype(img.dtype)

    def write_images(self, directory, epsilon=1e-6):
        """Apply shading correction and save corrected images to directory.

        Parameters
        ----------
        directory : str or Path
            Output directory for corrected images
        epsilon : float, optional
            Numerical stability parameter (default: 1e-6)
        """
        # Create the output directory
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Loop over the images
        for i in tqdm.tqdm(range(self.n_images), desc="Shading Correction"):
            this_file = self.files[i]
            this_img = self.img_stack[i, ...]

            # Normalize the image
            img_p = self.normalize(this_img, epsilon=epsilon)

            # Get the output filename
            filename = directory / this_file.name

            # Save the file
            cv2.imwrite(str(filename), img_p)

    def prepare(self, img_stack=None):
        """Prepare optimizer for fitting.

        Loads images, resamples, initializes regularization parameters and
        variables for optimization.

        Parameters
        ----------
        img_stack : np.ndarray, optional
            Pre-loaded image stack. If None, loads from configured source (default: None)
        """
        # Load the data
        if img_stack is not None:
            self._load_images(img_stack)
        elif self.input_type in ["directory", "files_list"]:
            self._load_images()
        elif self.input_type in ["images_stack", "images_list"]:
            self._load_images(self.img_stack)

        # Initialize the regularization parameters
        mean_value = self.img_stack_resized.mean(axis=0)
        mean_value /= mean_value.mean() # Normalized pixel-wise mean of all images
        mean_value_dct = dctn(mean_value, norm='ortho')

        if self.l_s is None:
            self.l_s = np.abs(mean_value_dct).sum() / 800.0
        if self.l_d is None:
            self.l_d = np.abs(mean_value_dct).sum() / 2000.0

        # Construct the measurement matrix
        self.img_sort = np.sort(self.img_stack_resized, axis=0)

        # Initialize the darkfield, flatfield, offset, and weights (for the L1 reweighted loss)
        self.offset = np.zeros([self.working_size]*2)
        self.flatfield = np.ones([self.working_size]*2)
        self.flatfield_fullsize = np.ones(self.image_shape)
        self.darkfield = np.random.randn(self.working_size, self.working_size)
        self.darkfield_fullsize = np.zeros(self.image_shape)
        self.W = np.ones_like(self.img_sort)

        # Initialize other parameters
        self.iteration = 0
        self.flag_reweigthing = True

    def update_weights(self):
        """Update weighting matrix for iterative reweighted L1-norm optimization.

        Computes weights based on residuals for robust estimation.
        """
        # Weight Update formula in the paper
        # self.W = 1.0 / (np.abs(self.Ir / self.Ib) + self.epsilon)

        # Actual Weight update formula in the matlab implementation
        self.W = 1.0 / (np.abs(self.Ir / (self.Ib.mean() + 1e-6)) + self.epsilon)
        self.W = self.W * self.W.size / self.W.sum()
        self.reweighting_iteration += 1

    def update(self):
        """Perform one optimization iteration.

        Executes LADM optimization and updates flatfield, darkfield, and weights.
        Checks convergence and manages reweighting iterations.
        """
        last_flatfield = self.flatfield.copy()
        last_darkfield = self.darkfield.copy()

        # Perform LADM optimization
        Ib, Ir, D = inexact_alm_l1(self.img_sort, self.l_s, self.l_d, weight=self.W, estimateDarkField=self.estimate_darkfield, verbose=self.verbose)

        # Reshape the images.
        self.Ib = np.reshape(Ib, (self.n_images, self.working_size, self.working_size)) # Flat-field
        self.Ir = np.reshape(Ir, (self.n_images, self.working_size, self.working_size)) # Residual
        D = np.reshape(D, (self.working_size, self.working_size)) # Dark-field

        # Update the weight matrix
        self.update_weights()

        # Update the flat-field and dark-field
        self.flatfield = self.Ib.mean(axis=0) - D
        self.flatfield = self.flatfield / self.flatfield.mean()
        self.darkfield = D

        # Compute the difference between the new fields and the last ones.
        mad_flatfield = np.abs(self.flatfield - last_flatfield).sum() / np.abs(last_flatfield).sum()
        mad_darkfield = np.abs(self.darkfield - last_darkfield).sum()
        if mad_darkfield < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = mad_darkfield / max(np.abs(last_darkfield).sum(), 1e-6)

        # Check if another L1 reweighting is necessary
        if (max(mad_flatfield, mad_darkfield)<=self.reweighting_tolerance) or (self.reweighting_iteration > self.max_reweightingIterations):
            self.flag_reweigthing = False

    def run(self):
        """Execute the BaSiC optimization algorithm.

        Performs iterative reweighting until convergence or max iterations.
        Upsamples final flatfield and darkfield to original image dimensions.
        """
        if self.verbose:
            pbar = tqdm.tqdm(desc="Reweighting Iteration", total=self.max_reweightingIterations)
        while self.flag_reweigthing:
            self.update()
            if self.verbose:
                pbar.update()
            # self.display_fields()
        if self.verbose:
            pbar.close()

        # Reshape the flat and dark fields to the original shape
        self.flatfield_fullsize = cv2.resize(self.flatfield.T, self.image_shape, cv2.INTER_LINEAR).T
        self.flatfield_fullsize = self.flatfield_fullsize / self.flatfield_fullsize.mean()
        self.darkfield_fullsize = cv2.resize(self.darkfield.T, self.image_shape, cv2.INTER_LINEAR).T

    def set_flatfield(self, flatfield):
        """Set a custom flatfield correction.

        Parameters
        ----------
        flatfield : np.ndarray
            Flatfield image to use
        """
        self.flatfield_fullsize = cv2.resize(flatfield.T, self.image_shape, cv2.INTER_LINEAR).T

    def set_darkfield(self, darkfield):
        """Set a custom darkfield correction.

        Parameters
        ----------
        darkfield : np.ndarray
            Darkfield image to use
        """
        self.darkfield_fullsize = cv2.resize(darkfield.T, self.image_shape, cv2.INTER_LINEAR).T

    def get_flatfield(self):
        """Get the estimated or set flatfield correction.

        Returns
        -------
        np.ndarray
            Flatfield correction image
        """
        return self.flatfield_fullsize.copy()

    def get_darkfield(self):
        """Get the estimated or set darkfield correction.

        Returns
        -------
        np.ndarray
            Darkfield correction image
        """
        return self.darkfield_fullsize.copy()

def shrink(theta, epsilon=1e-3):
    """Scalar shrinkage operator (soft-thresholding).

    Applies soft-thresholding to input values for L1-norm minimization.

    Parameters
    ----------
    theta : np.ndarray or float
        Input values
    epsilon : float, optional
        Threshold value (default: 1e-3)

    Returns
    -------
    np.ndarray or float
        Shrunk values
    """
    theta_p = np.sign(theta) * np.maximum(np.abs(theta) - epsilon, 0)
    return theta_p

def inexact_alm_l1(imgs, l_s, l_d, tol=1e-6, maxIter=500, weight=1, estimateDarkField=True, rho=1.5, verbose=False):
    """Inexact augmented Lagrangian method for sparse low-rank recovery.

    Solves the BaSiC L1-minimization problem to decompose image stack into
    flatfield, darkfield, and residual components.

    Parameters
    ----------
    imgs : np.ndarray
        Image stack of shape (N_images * working_size, working_size)
    l_s : float
        Flatfield regularization parameter
    l_d : float
        Dark-field regularization parameter
    tol : float
        Convergence tolerance
    maxIter : int
        Maximum iterations number
    weight : N x P x Q ndarray
        Optional weight matrix used for the reweighted L1 norm
    estimateDarkField : bool
        Set to True to estimate the darkfield in addition to the flat field
    darkFieldLimit : float
        Maximum value for the darkfield, use to constrain the minimization
    rho : float
        Lagrange multiplier learning rate

    Returns
    -------
    S : ndarray
        Estimated unnormalised flat-field
    Ib : ndarray

    Ir : ndarray



    Notes
    -----
    % modified from the BaSiC matlab implementation
    % modified from Robust PCA
    % reference:
    % Peng et al. "A BaSiC tool for background and shading correction
    % of optical microscopy images" Nature Communications, 14836(2017)
    % Candès, E., Li, X., Ma, Y. & Wright, J. "Robust Principal Component
    % Analysis?" J. ACM (58) 2011

    % D - m x m x n matrix of observations/data (required input)
    %

    % while ~converged
    %   minimize (inexactly, update A and E only once)
    %   L(W, E,Y,u) = |E|_1+lambda * |W|_1 + <Y2,D-repmat(QWQ^T)-E> + +mu/2 * |D-repmat(QWQ^T)-E|_F^2;
    %   Y1 = Y1 + \mu * (D - repmat(QWQ^T) - E);
    %   \mu = \rho * \mu;
    % end
    %
    % Tingying Peng (tingying.peng@tum.de)

    %
    % Copyright: CAMP, Technical University of Munich

    """
    ###############################
    # Initialize the optimization #
    ###############################
    N, P, Q = imgs.shape[:]

    # Reshape the image stack as a measurement matrix
    D = np.reshape(imgs, (N, P*Q))
    d_norm = np.linalg.norm(D, "fro")
    W = np.reshape(weight, D.shape)
    B1_uplimit = D.min()
    B1 = 0

    # Initialize Ib, Ir, B, S and D
    S = np.zeros_like(D)  # Flat-field
    Sf = dctn(np.reshape(S, (N,P,Q)).mean(axis=0), norm='ortho') # Flat-field, in Fourier Domain
    Ir = np.zeros_like(D)  # Residual
    B = np.ones((N,1)) # Image baseline
    D_field = np.zeros((1, P*Q)) # Darkfield

    # Initialize Lagrange multiplier
    Y = 0
    D_svd = np.linalg.svd(D, compute_uv=False)
    mu = 12.5 / D_svd[0] # TODO: This can be tuned
    mu_bar = mu * 1e7
    ent2 = 10

    converged = False
    iteration = 0
    if verbose:
        pbar = tqdm.tqdm(desc="Iteration", total=maxIter)
    while not(converged) and (iteration < maxIter): # TODO: Add tqdm
        # 1. Compute DCT of the flat-field Sf(k), and Ib(k)
        S = np.reshape(idctn(Sf, norm='ortho'), (1,P*Q)) # Flat-field in spatial domain
        Ib = S * B + D_field # with broadcasting, as S (NxPQ), B (Nx1), D(1xPQ)

        # 2. Update Sf(k) -> Sf(k+1)
        dS = (D - Ib - Ir + Y / mu) # Shape: N x PQ
        dS = np.reshape(dS, (N, P, Q))
        dS = dS.mean(axis=0) # Averate over all images
        Sf = Sf + dctn(dS, norm='ortho')
        Sf = shrink(Sf, l_s / mu) # Scalar shrinkage operator

        # 3. Get S(k+1,x) and mid-iteration Ib(k+1/2)
        S = np.reshape(idctn(Sf, norm='ortho'), (1, P*Q)) # Flat-field at iteration S(k+1)
        Ib = S * B + D_field # Ib, mid iteration

        # 4. Update the residual Ir(k) -> Ir(k+1)
        dIr = (D - Ib - Ir + Y / mu)
        Ir = Ir + dIr
        Ir = shrink(Ir, W/mu)

        # 5. Update the image baseline B(k) -> B(k+1), and Ib(k+1)
        R = D - Ir
        B = R.mean(axis=1, keepdims=True) / R.mean()
        B[B < 0] = 0 # Enforce non-negativity constraint
        # Ib = S * B + D_field

        # Dark-field optimization
        if estimateDarkField: # TODO: Add the darkfield optimization here
            # 1. Dz estimation with least-mean square
            mask_validB = B < 1
            mask_highS = S > (S.mean() - 1e-6)
            mask_lowS = S < (S.mean() + 1e-6)
            R_high = np.mean(R * (mask_highS * mask_validB), axis=1, keepdims=True)
            R_low = np.mean(R * (mask_lowS * mask_validB), axis=1, keepdims=True)
            B1 = (R_high - R_low)/R.mean()
            k = mask_validB.sum()
            temp1 = np.sum(B[mask_validB]**2)
            temp2 = B[mask_validB].sum()
            temp3 = B1.sum()
            temp4 = np.sum(B[mask_validB] * B1)
            temp5 = temp2 * temp3 - k * temp4
            if temp5 == 0:
                B1 = 0
            else:
                B1 = (temp1*temp3 - temp2*temp4) / temp5

            # Clipping to limit the range of Dz to 0 and min(D)
            B1 = np.maximum(B1, 0) # Non negativity constraing
            B1 = np.minimum(B1, B1_uplimit / S.mean())

            # 2. Dr optimization
            Z = B1 * (S.mean() - S)

            A1_offset = np.ma.masked_array(R, np.tile(~mask_validB, (1, P*Q))).mean(axis=0, keepdims=True) - B[mask_validB].mean() * S
            A1_offset = A1_offset - A1_offset.mean()
            A_offset = A1_offset - A1_offset.mean() - Z

            # Smooth A_offset (Dr)
            Dr_f = dctn(np.reshape(A_offset, (P,Q)), norm='ortho') # Fourier-domain flat-field residual
            Dr_f = shrink(Dr_f, l_d / (ent2 * mu)) # Shrink operator in Fourier-domain for smooth residual
            Dr = idctn(Dr_f, norm='ortho').reshape((1, P*Q))
            Dr = shrink(Dr, l_d / (mu*ent2)) # Shrink operator in spatial domain for sparse redisual
            D_field = Dr + Z

        # 6. Update the Lagrange multiplier Y(k)->Y(k+1), mu(k)->mu(k+1), and k->k+1
        dY = D - Ib - Ir
        Y = Y + mu * dY
        mu = min(mu * rho, mu_bar) # Clip mu
        iteration += 1

        # Evaluate convergence
        stopCriterion = np.linalg.norm(dY, "fro") / d_norm
        if verbose:
            pbar.update()
        # pbar.set_description(f"Iteration (Criterion={stopCriterion:.3e})", refresh=False)
        if stopCriterion < tol:
            converged = True

        # TODO: Add logging

    if iteration == maxIter:
        print("Maximum iterations reached")
    if verbose:
        pbar.close()
    # Update the darkfield, with
    D_field = D_field + B1*S

    return Ib, Ir, D_field










@njit
def mean_axis_0(arr):
    """Compute mean along axis 0 using numba JIT compilation.

    Parameters
    ----------
    arr : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Mean values along axis 0
    """
    return np.array([arr[..., i].mean() for i in range(arr.shape[1])])

def get_photobleach(imgflt_stack, flatfield, darkfield=None, size=(128,128)):
    """Estimate photobleaching coefficients from image stack.

    Analyzes temporal intensity changes to estimate photobleaching effects
    across image frames. Useful for determining frame-wise illumination decay.

    Parameters
    ----------
    imgflt_stack : np.ndarray
        Stack of image frames
    flatfield : np.ndarray
        Flatfield correction image
    darkfield : np.ndarray, optional
        Darkfield correction image (default: None)
    size : tuple, optional
        Working size (x, y) for computation (default: (128, 128))

    Returns
    -------
    np.ndarray
        Photobleaching coefficients for each frame
    """
    # Size is (x,y)
    img_stack = np.zeros((size[1], size[0], imgflt_stack.shape[0]))

    for i, frame in enumerate(imgflt_stack):
        img_stack[:, :, i] = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR).astype(np.float64)

    imgflt_stack = img_stack

    flatfield_small = cv2.resize(flatfield, size, interpolation=cv2.INTER_LINEAR).astype(np.float64)
    if darkfield is not None:
        darkfield_small = cv2.resize(darkfield, size, interpolation=cv2.INTER_LINEAR).astype(np.float64)

    imgflt_stack = np.reshape(imgflt_stack, (size[0] * size[1], -1)).astype(np.float64)

    imgflt_stack_svd = np.linalg.svd(imgflt_stack, full_matrices=False, compute_uv=False)

    return _get_photobleach(imgflt_stack, imgflt_stack_svd, flatfield_small, darkfield_small)

# From https://github.com/PolusAI/polus-plugins/tree/dev/regression/polus-basic-flatfield-correction-plugin
# ------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2019 LabShare
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

@njit
def _get_photobleach(imgflt_stack, imgflt_stack_svd, flatfield, darkfield=None):
    """Calculate the global effect of photobleaching for each image
    Using the original data, flatfield, and darkfield images, estimate the total
    contribution of photobleaching to an image in a series of images.
    Inputs:
        imgflt_stack - Numpy stack of images
        flatfield - numpy floating precision matrix containing flatfield values
        darkfield - numpy floating precision matrix containing darkfield values
    Outputs:
        A_coeff - A 1xn matrix of photobleaching offsets, where n is the number
            of input images
    """


    # Initialize matrices
    # Initialize matrices
    # imgflt_stack = np.reshape(
    #     imgflt_stack, (size * size, -1)
    # ).astype(np.float64)
    if darkfield is None:
        darkfield = np.zeros(flatfield.shape, dtype=np.float64)

    # Initialize weights and tolerances
    weights = np.ones(imgflt_stack.shape, dtype=np.float64)
    epsilon = np.float64(0.1)
    tol = np.float64(10 ** -6)

    # Run optimization exactly 5 times
    for r in range(5):
        # print(r)
        # Calculate weights, offsets and coefficients
        W_idct_hat = np.reshape(flatfield, (-1, 1))
        A_offset = np.reshape(darkfield, (-1, 1))
        A_coeff = np.reshape(mean_axis_0(imgflt_stack), (1, -1))

        # Initialization values and learning rates
        # temp = np.linalg.svd(imgflt_stack, full_matrices=False, compute_uv=False)
        temp = imgflt_stack_svd
        norm_two = np.float64(temp[0])
        mu = np.float64(12.5) / norm_two
        mu_bar = mu * 10 ** 7
        rho = np.float64(1.5)

        ent1 = 1

        # Normalization factors
        d_norm = np.linalg.norm(imgflt_stack)#, "fro")

        # Initialize augmented representation and error
        A = np.zeros(imgflt_stack.shape, dtype=np.float64)
        E1 = np.zeros(imgflt_stack.shape, dtype=np.float64)
        # Y1 = np.float64(0)
        Y1 = np.zeros(imgflt_stack.shape)

        # Run optimization
        iternum = 0
        converged = False
        while not converged:
            iternum += 1
            # print(iternum)
            # Calculate augmented representation
            # A = np.matmul(W_idct_hat, A_coeff) + A_offset
            A = (W_idct_hat * A_coeff) + A_offset

            # Calculate errors
            # E1 = E1 + np.divide(imgflt_stack - A - E1 + np.multiply(1 / mu, Y1), ent1)
            E1 = imgflt_stack - A + Y1 / mu

            # E1 = np.max(
            #     np.reshape(
            #         E1 - weights / (ent1 * mu),
            #         (imgflt_stack.shape[0], imgflt_stack.shape[1], 1),
            #     ),
            #     -1,
            #     initial=10 ** -6,
            # ) + np.min(
            #     np.reshape(
            #         E1 + weights / (ent1 * mu),
            #         (imgflt_stack.shape[0], imgflt_stack.shape[1], 1),
            #     ),
            #     -1,
            #     initial=0,
            # )

            wem = weights / (ent1 * mu)
            E1 = np.maximum(E1 - wem, 10 ** -6) + np.minimum(E1 + wem, 0)

            # Calculate coefficients
            R1 = imgflt_stack - E1
            # A_coeff = np.reshape(np.mean(R1, axis=0), (1, -1)) - np.mean(A_offset)
            A_coeff = np.reshape(mean_axis_0(R1), (1, -1)) - np.mean(A_offset)
            # A_coeff[A_coeff < 0] = 0  # pixel values are never negative
            A_coeff = np.maximum(A_coeff, 0)

            # Loss
            Z1 = imgflt_stack - A - E1

            # Error updates
            Y1 = Y1 + mu * Z1

            # Update learning rate
            # mu = np.min(mu * rho, initial=mu_bar)
            mu = np.minimum(mu * rho, mu_bar)

            # Stop if below threshold
            #stopCriterion = np.linalg.norm(Z1, "fro") / d_norm
            stopCriterion = np.linalg.norm(Z1) / d_norm

            # print(stopCriterion)
            if stopCriterion < tol:
                converged = True

        # Update weights
        # XE_norm = np.reshape(np.mean(A, axis=0), (1, -1)) / E1
        XE_norm = np.reshape( mean_axis_0(A), (1, -1)) / E1
        weights = 1 / np.abs(XE_norm + epsilon)
        weights = weights * weights.size / np.sum(weights)

    return A_coeff


#
# #original
#
# OPTIONS = {'size':128}
# def _get_photobleach(imgflt_stack, flatfield, darkfield=None):
#     """Calculate the global effect of photobleaching for each image
#     Using the original data, flatfield, and darkfield images, estimate the total
#     contribution of photobleaching to an image in a series of images.
#     Inputs:
#         imgflt_stack - Numpy stack of images
#         flatfield - numpy floating precision matrix containing flatfield values
#         darkfield - numpy floating precision matrix containing darkfield values
#     Outputs:
#         A_coeff - A 1xn matrix of photobleaching offsets, where n is the number
#             of input images
#     """
#     # Initialize matrices
#     imgflt_stack = np.reshape(
#         imgflt_stack, (OPTIONS["size"] * OPTIONS["size"], -1)
#     ).astype(np.float64)
#     if darkfield is None:
#         darkfield = np.zeros(flatfield.shape, dtype=np.float64)
#
#     # Initialize weights and tolerances
#     weights = np.ones(imgflt_stack.shape, dtype=np.float64)
#     epsilon = np.float64(0.1)
#     tol = np.float64(10 ** -6)
#
#     # Run optimization exactly 5 times
#     for r in range(5):
#         # Calculate weights, offsets and coefficients
#         W_idct_hat = np.reshape(flatfield, (-1, 1))
#         A_offset = np.reshape(darkfield, (-1, 1))
#         A_coeff = np.reshape(np.mean(imgflt_stack, 0), (1, -1))
#
#         # Initialization values and learning rates
#         temp = np.linalg.svd(imgflt_stack, full_matrices=False, compute_uv=False)
#         norm_two = np.float64(temp[0])
#         mu = np.float64(12.5) / norm_two
#         mu_bar = mu * 10 ** 7
#         rho = np.float64(1.5)
#         ent1 = 1
#
#         # Normalization factors
#         d_norm = np.linalg.norm(imgflt_stack, "fro")
#
#         # Initialize augmented representation and error
#         A = np.zeros(imgflt_stack.shape, dtype=np.float64)
#         E1 = np.zeros(imgflt_stack.shape, dtype=np.float64)
#         Y1 = np.float64(0)
#
#         # Run optimization
#         iternum = 0
#         converged = False
#         while not converged:
#             iternum += 1
#
#             # Calculate augmented representation
#             A = np.matmul(W_idct_hat, A_coeff) + A_offset
#
#             # Calculate errors
#             E1 = E1 + np.divide(imgflt_stack - A - E1 + np.multiply(1 / mu, Y1), ent1)
#             E1 = np.max(
#                 np.reshape(
#                     E1 - weights / (ent1 * mu),
#                     (imgflt_stack.shape[0], imgflt_stack.shape[1], 1),
#                 ),
#                 -1,
#                 initial=10 ** -6,
#             ) + np.min(
#                 np.reshape(
#                     E1 + weights / (ent1 * mu),
#                     (imgflt_stack.shape[0], imgflt_stack.shape[1], 1),
#                 ),
#                 -1,
#                 initial=0,
#             )
#
#             # Calculate coefficients
#             R1 = imgflt_stack - E1
#             A_coeff = np.reshape(np.mean(R1, 0), (1, -1)) - np.mean(A_offset)
#             A_coeff[A_coeff < 0] = 0  # pixel values are never negative
#
#             # Loss
#             Z1 = imgflt_stack - A - E1
#
#             # Error updates
#             Y1 = Y1 + mu * Z1
#
#             # Update learning rate
#             mu = np.min(mu * rho, initial=mu_bar)
#
#             # Stop if below threshold
#             stopCriterion = np.linalg.norm(Z1, "fro") / d_norm
#             if stopCriterion < tol:
#                 converged = True
#
#         # Update weights
#         XE_norm = np.reshape(np.mean(A, 0), (1, -1)) / E1
#         weights = 1 / np.abs(XE_norm + epsilon)
#         weights = weights * weights.size / np.sum(weights)
#
#     return A_coeff


# Implemented based on Matlab code https://github.com/marrlab/BaSiC
# def BaSiC_basefluor(IF, flatfield, darkfield=None):
#     img_stack = IF
#     nrows = 128
#     ncols = 128
#     working_size = 128
#     n_images = img_stack.shape[0]
#
#
#     image_shape = img_stack.shape[1::]
#     new_shape = tuple([working_size] * 2)
#     img_stack_p = np.zeros([n_images, *new_shape], dtype=img_stack.dtype)
#     if working_size > image_shape[0]:
#         interpolation = cv2.INTER_LINEAR
#     else:
#         interpolation = cv2.INTER_AREA
#     for i in range(n_images):
#         img = img_stack[i, ...].squeeze()
#         img_stack_p[i, ...] = cv2.resize(img.T, new_shape, interpolation=interpolation).T
#     img_stack_resized = img_stack_p.astype(np.float32)
#     D = img_stack_resized.reshape(-1,nrows*ncols)
#
#
#     flatfield = cv2.resize(flatfield.T, new_shape, interpolation=interpolation).T
#     if darkfield is not None:
#         darkfield = cv2.resize(darkfield.T, new_shape, interpolation=interpolation).T
#
#     weight = np.ones(D.shape)
#     eplson = 0.1
#     tol = 1e-6
#
#     for reweighting_iter in range(5):
#         W_idct_hat = flatfield.flatten()
#         A_offset = darkfield.flatten()
#         A1_coeff = np.mean(D, 0)
#         # main iteration loop starts
#         temp = np.linalg.svd(D, compute_uv=False)
#         norm_two = temp[0]
#         mu = 12.5 / norm_two # this one can be tuned
#         mu_bar = mu * 1e7
#         rho = 1.5 # this one can be tuned
#         d_norm = np.linalg.norm(D, 'fro')
#         ent1 = 1
#         iter = 0
#         total_svd = 0
#         converged = False
#         A1_hat = np.zeros(D.shape)
#         E1_hat = np.zeros(D.shape)
#         Y1 = 0
#
#         while not converged:
#             iter = iter + 1
#             A1_hat = W_idct_hat * A1_coeff + A_offset
#
#             # update E1 using l0 norm
#             E1_hat = E1_hat + (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
#             E1_hat = np.maximum(E1_hat - weight / (ent1 * mu), 0) + np.minimum(E1_hat + weight / (ent1 * mu), 0)
#             # update A1_coeff, A2_coeff and A_offset
#             # if coeff_flag
#
#             # R1 = bsxfun( @ minus, D - E1_hat, A_offset)
#             R1 = D - E1_hat - A_offset
#             A1_coeff = np.mean(R1,0) - np.mean(A_offset)
#             # [A1_mincoeff, A1_mincoefftile] = min(A1_coeff);
#             # A1_back(A1_coeff < 1) = 1;
#             A1_coeff[A1_coeff < 0] = 0
#
#             # A_offset = max(min(A_offset, A_uplimit), 0);
#             Z1 = D - A1_hat - E1_hat
#             # Z2 = D - A2_hat - E2_hat;
#
#             Y1 = Y1 + mu * Z1
#             # Y2 = Y2 + mu * Z2;
#
#             mu = np.minimum(mu * rho, mu_bar)
#
#             # stop Criterion
#             stopCriterion = np.linalg.norm(Z1, 'fro') / d_norm
#             # stopCriterion = max(norm(Z1, 'fro') / d_norm, norm(Z2, 'fro') / d_norm);
#             if stopCriterion < tol:
#                 converged = True
#
#             if total_svd % 10 == 0:
#                 print('Iteration ' + str(iter) + ' |E1|_0 ' + str(np.sum(np.abs(E1_hat.flatten()) > 0)) + \
#                 ' stopCriterion ' + str(stopCriterion))
#
#         # update weight
#         # XE_norm = bsxfun( @ ldivide, E1_hat, mean(A1_hat));
#         XE_norm = np.mean(A1_hat) / E1_hat
#         weight = 1 / (np.abs(XE_norm) + eplson)
#
#         # % if isempty(segmentation)
#         #     % weight(segmentation
#         #     ~ = 0)=0;
#         # % end
#         weight = weight * weight.size / weight.sum()
#
#     return A1_coeff
#


