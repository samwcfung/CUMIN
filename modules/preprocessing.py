#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module
------------------
Implements image preprocessing methods including:
- Photobleaching correction (polynomial detrending, exponential fits, etc.)
- Background removal (uniform filtering, morphological tophat)
- Denoising (gaussian, anisotropic, median, bilateral)
- Stripe correction
- CNMF-based methods

Utilizes GPU acceleration via CuPy when available for supported methods.
"""

import os
import time
import numpy as np
import h5py
import tifffile
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.optimize import curve_fit
from scipy import signal
import cv2

# Try to import CuPy for GPU acceleration, fallback to CPU if not available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not found, falling back to CPU processing for applicable methods")

# Try to import xarray for DataArray support
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    warnings.warn("xarray not found, some functions will be unavailable")

# Try to import medpy for anisotropic diffusion
try:
    from medpy.filter.smoothing import anisotropic_diffusion
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False
    warnings.warn("medpy not found, anisotropic diffusion will be unavailable")

# Try to import skimage for morphological operations
try:
    from skimage.transform import resize
    from skimage.morphology import disk
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("skimage not found, some morphological operations will be unavailable")


def preprocess_for_cnmfe(data, spatial_downsample=2, temporal_downsample=1, use_float16=True, logger=None):
    """
    Preprocess data for CNMF-E to reduce memory usage and computation time.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    spatial_downsample : int
        Factor to downsample spatially (2 = half size)
    temporal_downsample : int
        Factor to downsample temporally (2 = half frames)
    use_float16 : bool
        Whether to convert to float16 to save memory
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Preprocessed data
    """
    if not HAS_SKIMAGE:
        raise ImportError("skimage.transform is required for spatial downsampling")
    
    start_time = time.time()
    
    if logger:
        logger.info(f"Preprocessing data for CNMF-E: {data.shape}")
        logger.info(f"Spatial downsample: {spatial_downsample}x, Temporal downsample: {temporal_downsample}x")
        logger.info(f"Using float16: {use_float16}")
    
    n_frames, height, width = data.shape
    
    # Skip if no downsampling and no type conversion
    if spatial_downsample == 1 and temporal_downsample == 1 and not use_float16:
        if logger:
            logger.info("No preprocessing needed")
        return data
    
    # Apply temporal downsampling
    if temporal_downsample > 1:
        if logger:
            logger.info(f"Downsampling temporally by {temporal_downsample}x")
        
        # Select every n-th frame
        data = data[::temporal_downsample]
        n_frames = data.shape[0]
        
        if logger:
            logger.info(f"New frame count: {n_frames}")
    
    # Apply spatial downsampling
    if spatial_downsample > 1:
        if logger:
            logger.info(f"Downsampling spatially by {spatial_downsample}x")
        
        new_height = height // spatial_downsample
        new_width = width // spatial_downsample
        
        # Resize each frame
        resized_data = np.zeros((n_frames, new_height, new_width), 
                               dtype=np.float16 if use_float16 else np.float32)
        
        for i in range(n_frames):
            resized_data[i] = resize(data[i], (new_height, new_width), 
                                    preserve_range=True, anti_aliasing=True)
            
            # Log progress periodically
            if logger and i % 100 == 0:
                logger.info(f"Resized {i}/{n_frames} frames")
        
        data = resized_data
        
        if logger:
            logger.info(f"New dimensions: {data.shape}")
    
    # Convert to float16 if requested
    if use_float16 and data.dtype != np.float16:
        if logger:
            logger.info("Converting to float16 to save memory")
        data = data.astype(np.float16)
    
    if logger:
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        memory_usage_mb = data.nbytes / (1024 * 1024)
        logger.info(f"Data size: {memory_usage_mb:.2f} MB")
    
    return data


def get_cnmf_components(cnm, logger=None):
    """
    Safely extract components from a CNMF/CNMF-E object, handling different CaImAn versions.
    
    Parameters
    ----------
    cnm : CNMF or CNMFE object
        The fitted model object
    logger : logging.Logger, optional
        Logger for messages
        
    Returns
    -------
    dict
        Dictionary of components (A, C, b, f, S, etc.)
    """
    components = {}
    
    try:
        # Handle different CaImAn versions
        if hasattr(cnm, 'A') and hasattr(cnm, 'C'):
            # Old CaImAn version
            components['A'] = cnm.A
            components['C'] = cnm.C
            
            if hasattr(cnm, 'b'):
                components['b'] = cnm.b
            if hasattr(cnm, 'f'):
                components['f'] = cnm.f
            if hasattr(cnm, 'S'):
                components['S'] = cnm.S
            if hasattr(cnm, 'YrA'):
                components['YrA'] = cnm.YrA
                
        elif hasattr(cnm, 'estimates'):
            # New CaImAn version (post 1.8)
            est = cnm.estimates
            if hasattr(est, 'A'):
                components['A'] = est.A
            if hasattr(est, 'C'):
                components['C'] = est.C
            if hasattr(est, 'b'):
                components['b'] = est.b
            if hasattr(est, 'f'):
                components['f'] = est.f
            if hasattr(est, 'S'):
                components['S'] = est.S
            if hasattr(est, 'YrA'):
                components['YrA'] = est.YrA
            if hasattr(est, 'idx_components'):
                components['idx_components'] = est.idx_components
            if hasattr(est, 'idx_components_bad'):
                components['idx_components_bad'] = est.idx_components_bad
        
        if logger:
            logger.info(f"Successfully extracted components: {list(components.keys())}")
    
    except Exception as e:
        if logger:
            logger.warning(f"Error extracting CNMF components: {str(e)}")
    
    return components


def safe_hdf5_value(value):
    """
    Convert a value to a type that can be safely stored in HDF5.
    
    Parameters
    ----------
    value : any
        The value to convert
        
    Returns
    -------
    any
        A HDF5-compatible version of the value
    """
    if value is None:
        return -1  # Use -1 to represent None
    elif isinstance(value, (bool, int, float, str)):
        return value  # These types are safe
    elif isinstance(value, np.ndarray):
        if value.dtype == np.dtype('O'):
            # Convert object arrays to strings
            return str(value.tolist())
        else:
            return value  # Numeric arrays are safe
    elif isinstance(value, (list, tuple)):
        try:
            # Try to convert to a numeric array
            arr = np.array(value)
            if arr.dtype == np.dtype('O'):
                # If it's still an object array, convert to string
                return str(value)
            return arr
        except:
            # If conversion fails, use string representation
            return str(value)
    elif isinstance(value, dict):
        # Convert dict to string representation
        return str(value)
    else:
        # For any other type, convert to string
        return str(value)


def detrend_movie(data, detrend_degree=1, smoothing_sigma=2.0, max_frame=None, logger=None):
    """
    Detrend a movie to account for photobleaching using polynomial detrending.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    detrend_degree : int
        Degree of polynomial for detrending (1=linear, >1=nth degree polynomial)
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    max_frame : int, optional
        Maximum frame to process
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Detrended movie data
    """
    if logger:
        logger.info(f"Detrending movie with polynomial degree {detrend_degree}")
    
    # Get dimensions
    n_frames, height, width = data.shape
    
    # Limit to max_frame if specified
    if max_frame is not None and max_frame < n_frames:
        n_frames = max_frame
        data = data[:n_frames]
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    
    # Timepoints (frame indices)
    t = np.arange(n_frames)
    
    # Fit polynomial to mean fluorescence values
    poly_coeffs = np.polyfit(t, mean_f, detrend_degree)
    fitted_trend = np.polyval(poly_coeffs, t)
    
    # Calculate global mean across all pixels and frames
    global_mean = float(np.mean(data))
    
    if logger:
        logger.info(f"Fitted {detrend_degree}-order polynomial for detrending")
        logger.info(f"Global mean intensity: {global_mean:.4f}")
    
    # Apply correction
    corrected_data = np.zeros_like(data)
    
    for i in range(n_frames):
        # Divide by trend and multiply by global mean
        # Avoid division by zero
        corrected_data[i] = data[i] * (global_mean / max(fitted_trend[i], 1e-6))
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data


def exponential_decay_correction(data, smoothing_sigma=2.0, logger=None):
    """
    Correct photobleaching using an exponential decay model.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Corrected data
    """
    if logger:
        logger.info("Applying exponential decay correction")
    
    n_frames, height, width = data.shape
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    t = np.arange(n_frames)
    
    # Define exponential decay function
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    # Initial parameter estimates
    p0 = [mean_f[0] - mean_f[-1], 1/n_frames, mean_f[-1]]
    
    try:
        # Fit the exponential decay model
        popt, _ = curve_fit(exp_decay, t, mean_f, p0=p0, maxfev=10000)
        fitted_trend = exp_decay(t, *popt)
        
        if logger:
            logger.info(f"Exponential fit parameters: a={popt[0]:.2f}, b={popt[1]:.6f}, c={popt[2]:.2f}")
    except:
        # Fall back to a simple exponential model if fitting fails
        if logger:
            logger.warning("Curve fitting failed, using simple exponential model")
        b_est = -np.log(mean_f[-1]/mean_f[0]) / n_frames
        fitted_trend = mean_f[0] * np.exp(-b_est * t)
    
    # Calculate global mean
    global_mean = float(np.mean(data))
    
    # Apply correction
    corrected_data = np.zeros_like(data)
    for i in range(n_frames):
        # Avoid division by zero
        factor = global_mean / max(fitted_trend[i], 1e-6)
        corrected_data[i] = data[i] * factor
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data


def bi_exponential_correction(data, smoothing_sigma=2.0, logger=None):
    """
    Correct photobleaching using a bi-exponential decay model.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Corrected data
    """
    if logger:
        logger.info("Applying bi-exponential decay correction")
    
    n_frames, height, width = data.shape
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    t = np.arange(n_frames)
    
    # Define bi-exponential decay function
    def bi_exp_decay(t, a1, b1, a2, b2, c):
        return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t) + c
    
    # Initial parameter estimates
    intensity_drop = mean_f[0] - mean_f[-1]
    p0 = [intensity_drop * 0.7, 0.01, intensity_drop * 0.3, 0.001, mean_f[-1]]
    
    try:
        # Fit the bi-exponential decay model
        popt, _ = curve_fit(bi_exp_decay, t, mean_f, p0=p0, maxfev=10000, 
                           bounds=([0, 0, 0, 0, 0], [np.inf, 1, np.inf, 1, np.inf]))
        fitted_trend = bi_exp_decay(t, *popt)
        
        if logger:
            logger.info(f"Bi-exponential fit parameters: a1={popt[0]:.2f}, b1={popt[1]:.6f}, " + 
                       f"a2={popt[2]:.2f}, b2={popt[3]:.6f}, c={popt[4]:.2f}")
    except:
        # Fall back to polynomial fit if bi-exponential fitting fails
        if logger:
            logger.warning("Bi-exponential fitting failed, falling back to 3rd order polynomial")
        poly_coeffs = np.polyfit(t, mean_f, 3)
        fitted_trend = np.polyval(poly_coeffs, t)
    
    # Calculate global mean
    global_mean = float(np.mean(data))
    
    # Apply correction
    corrected_data = np.zeros_like(data)
    for i in range(n_frames):
        # Avoid division by zero
        factor = global_mean / max(fitted_trend[i], 1e-6)
        corrected_data[i] = data[i] * factor
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data


def adaptive_percentile_correction(data, window_size=101, percentile=10, smoothing_sigma=2.0, logger=None):
    """
    Correct photobleaching using a sliding window percentile approach.
    More adaptive to complex bleaching patterns.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    window_size : int
        Size of the sliding window (should be odd)
    percentile : float
        Percentile to use for estimating baseline (0-100)
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Corrected data
    """
    from scipy.signal import medfilt
    
    if logger:
        logger.info(f"Applying adaptive percentile correction (window={window_size}, percentile={percentile})")
    
    n_frames, height, width = data.shape
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    
    # Make sure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Calculate trend using sliding window percentile
    half_window = window_size // 2
    padded_mean = np.pad(mean_f, (half_window, half_window), mode='reflect')
    trend = np.zeros_like(mean_f)
    
    for i in range(n_frames):
        window = padded_mean[i:i+window_size]
        trend[i] = np.percentile(window, percentile)
    
    # Apply median filter to smooth the trend
    trend = medfilt(trend, min(15, window_size // 3 * 2 + 1))
    
    # Calculate global mean
    global_mean = float(np.mean(data))
    
    # Apply correction
    corrected_data = np.zeros_like(data)
    for i in range(n_frames):
        # Avoid division by zero
        factor = global_mean / max(trend[i], 1e-6)
        corrected_data[i] = data[i] * factor
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data


def two_stage_detrend(data, first_degree=1, second_degree=3, smoothing_sigma=2.0, logger=None):
    """
    Apply a two-stage detrending process:
    1. First apply a low-order polynomial fit to correct major trends
    2. Then apply a higher-order polynomial fit to handle more complex bleaching patterns
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    first_degree : int
        Degree of polynomial for first stage detrending
    second_degree : int
        Degree of polynomial for second stage detrending
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Detrended movie data
    """
    if logger:
        logger.info(f"Applying two-stage detrending (degrees {first_degree} and {second_degree})")
    
    # Get dimensions
    n_frames, height, width = data.shape
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # STAGE 1: First pass with low-order polynomial
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    t = np.arange(n_frames)
    
    # Fit first polynomial
    poly_coeffs1 = np.polyfit(t, mean_f, first_degree)
    fitted_trend1 = np.polyval(poly_coeffs1, t)
    
    # Calculate global mean
    global_mean = float(np.mean(data))
    
    # Apply first correction
    stage1_data = np.zeros_like(data)
    for i in range(n_frames):
        # Divide by trend and multiply by global mean
        # Avoid division by zero
        stage1_data[i] = data[i] * (global_mean / max(fitted_trend1[i], 1e-6))
    
    if logger:
        logger.info(f"Completed first stage detrending with degree {first_degree}")
    
    # STAGE 2: Second pass with higher-order polynomial
    # Calculate mean fluorescence after first correction
    mean_f2 = np.mean(stage1_data, axis=(1, 2))
    
    # Fit second polynomial to catch more complex patterns
    poly_coeffs2 = np.polyfit(t, mean_f2, second_degree)
    fitted_trend2 = np.polyval(poly_coeffs2, t)
    
    # Calculate new global mean
    global_mean2 = float(np.mean(stage1_data))
    
    # Apply second correction
    corrected_data = np.zeros_like(data)
    for i in range(n_frames):
        # Avoid division by zero
        corrected_data[i] = stage1_data[i] * (global_mean2 / max(fitted_trend2[i], 1e-6))
    
    if logger:
        logger.info(f"Completed second stage detrending with degree {second_degree}")
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data


def stripe_correction(data, reduce_dim="height", on="mean", logger=None):
    """
    Correct striping artifacts in the data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    reduce_dim : str
        Dimension along which to calculate the mean ("height" or "width")
    on : str
        Calculation method:
        - "mean": Use the mean across all frames
        - "max": Use the max projection across all frames
        - "perframe": Apply correction to each frame separately
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Stripe-corrected data
    """
    if logger:
        logger.info(f"Applying stripe correction (reduce_dim={reduce_dim}, on={on})")
    
    # Get dimensions
    n_frames, height, width = data.shape
    
    # Calculate the reference image based on 'on' parameter
    if on == "mean":
        if logger:
            logger.info("Using mean projection for stripe correction")
        reference = np.mean(data, axis=0)
    elif on == "max":
        if logger:
            logger.info("Using max projection for stripe correction")
        reference = np.max(data, axis=0)
    elif on == "perframe":
        # Process each frame separately
        if logger:
            logger.info("Applying per-frame stripe correction")
        
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # For each frame, calculate mean along the specified dimension
            if reduce_dim == "height":
                mean1d = np.mean(data[i], axis=0)  # Mean along height for each column
                corrected_data[i] = data[i] - mean1d[np.newaxis, :]
            else:  # reduce_dim == "width"
                mean1d = np.mean(data[i], axis=1)  # Mean along width for each row
                corrected_data[i] = data[i] - mean1d[:, np.newaxis]
            
            # Log progress periodically
            if logger and i % 100 == 0:
                logger.info(f"Processed {i}/{n_frames} frames")
        
        if logger:
            logger.info("Stripe correction completed")
        
        return corrected_data
    else:
        raise ValueError(f"Parameter 'on={on}' not understood")
    
    # Calculate the mean along the specified dimension
    if reduce_dim == "height":
        mean1d = np.mean(reference, axis=0)  # Mean along height for each column
        # Apply correction
        corrected_data = data - mean1d[np.newaxis, np.newaxis, :]
    else:  # reduce_dim == "width"
        mean1d = np.mean(reference, axis=1)  # Mean along width for each row
        # Apply correction
        corrected_data = data - mean1d[np.newaxis, :, np.newaxis]
    
    if logger:
        logger.info("Stripe correction completed")
    
    return corrected_data


def correct_photobleaching(input_data, output_path=None, config=None, logger=None, save_output=True):
    """
    Correct photobleaching in fluorescence imaging data.
    
    This function corrects photobleaching effects frame by frame by modeling the
    intensity decay over time and normalizing each frame. It can handle either
    numpy arrays or file paths, and save the results to an HDF5 file.
    
    Parameters
    ----------
    input_data : str or numpy.ndarray
        Either a path to a TIFF file or a numpy array with shape (frames, height, width)
    output_path : str, optional
        Path to save corrected data as HDF5. If None, data is only returned in memory.
    config : dict, optional
        Configuration dictionary with parameters for correction.
    logger : logging.Logger, optional
        Logger for status updates.
    save_output : bool, optional
        Whether to save the corrected data to disk. Default is True.
    
    Returns
    -------
    tuple
        (corrected_data, image_shape) tuple where corrected_data is a numpy array
        and image_shape is a tuple of (height, width)
    
    Notes
    -----
    Supported correction methods:
    - "polynomial_detrend": Fits a polynomial to mean intensities
    - "exponential_decay": Fits an exponential decay model
    - "bi_exponential": Fits a bi-exponential decay model
    - "adaptive_percentile": Uses a sliding window percentile approach
    - "two_stage_detrend": Applies two-stage polynomial detrending
    - "detrend_movie": Simple movie detrending
    - "lowpass_filter": Uses lowpass filter for correction
    """
    start_time = time.time()
    
    # Set default config if none provided
    if config is None:
        config = {
            "correction_method": "polynomial_detrend",
            "polynomial_order": 3,
            "smoothing_sigma": 2.0
        }
    
    # Extract parameters
    method = config.get("correction_method", "polynomial_detrend")
    
    if logger:
        logger.info(f"Starting photobleaching correction using {method} method")
    
    # Load data if input is a file path
    if isinstance(input_data, str):
        if logger:
            logger.info(f"Loading data from {input_data}")
        
        try:
            with tifffile.TiffFile(input_data) as tif:
                data = tif.asarray()
                
                # Check dimensions - some TIFFs may be in TZYX format
                if len(data.shape) > 3:
                    # Assume TZYX format and take first channel
                    if logger:
                        logger.info(f"Detected {len(data.shape)}D data, reshaping")
                    if len(data.shape) == 4:  # TZYX
                        data = data[:, 0] if data.shape[1] < data.shape[0] else data[0]
                    elif len(data.shape) == 5:  # TZCYX
                        data = data[:, 0, 0] if data.shape[1] < data.shape[0] else data[0, 0]
                
                # Ensure data is in (frames, height, width) format
                if len(data.shape) == 3:
                    if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                        # Already in (frames, height, width) format
                        pass
                    else:
                        # Try to rearrange to (frames, height, width)
                        if data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                            data = np.moveaxis(data, 2, 0)
                        elif data.shape[1] < data.shape[0] and data.shape[1] < data.shape[2]:
                            data = np.moveaxis(data, 1, 0)
                            
                if logger:
                    logger.info(f"Loaded data with shape {data.shape}")
        except Exception as e:
            if logger:
                logger.error(f"Error loading TIFF file: {str(e)}")
            raise
    else:
        # Input is already a numpy array
        data = input_data
    
    # Check input dimensions
    if len(data.shape) != 3:
        raise ValueError(f"Input data must have 3 dimensions (frames, height, width), got {data.shape}")
    
    # Get dimensions
    n_frames, height, width = data.shape
    image_shape = (height, width)
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Apply photobleaching correction based on selected method
    if method == "polynomial_detrend":
        polynomial_order = config.get("polynomial_order", 3)
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        
        if logger:
            logger.info(f"Using polynomial detrending with order {polynomial_order}")
        
        corrected_data = detrend_movie(
            data,
            detrend_degree=polynomial_order,
            smoothing_sigma=smoothing_sigma,
            logger=logger
        )
    
    elif method == "exponential_decay":
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        
        if logger:
            logger.info("Using exponential decay correction")
        
        corrected_data = exponential_decay_correction(
            data,
            smoothing_sigma=smoothing_sigma,
            logger=logger
        )
    
    elif method == "bi_exponential":
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        
        if logger:
            logger.info("Using bi-exponential decay correction")
        
        corrected_data = bi_exponential_correction(
            data,
            smoothing_sigma=smoothing_sigma,
            logger=logger
        )
    
    elif method == "adaptive_percentile":
        window_size = config.get("window_size", 101)
        percentile = config.get("percentile", 10)
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        
        if logger:
            logger.info(f"Using adaptive percentile correction (window={window_size}, percentile={percentile})")
        
        corrected_data = adaptive_percentile_correction(
            data,
            window_size=window_size,
            percentile=percentile,
            smoothing_sigma=smoothing_sigma,
            logger=logger
        )
    
    elif method == "two_stage_detrend":
        first_degree = config.get("first_degree", 1)
        second_degree = config.get("second_degree", 3)
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        
        if logger:
            logger.info(f"Using two-stage detrending (degrees {first_degree} and {second_degree})")
        
        corrected_data = two_stage_detrend(
            data,
            first_degree=first_degree,
            second_degree=second_degree,
            smoothing_sigma=smoothing_sigma,
            logger=logger
        )
        
    elif method == "detrend_movie":
        detrend_degree = config.get("polynomial_order", 1)
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        max_frame = config.get("max_frame", None)
        
        if logger:
            logger.info(f"Using simple detrending with degree {detrend_degree}")
        
        corrected_data = detrend_movie(
            data,
            detrend_degree=detrend_degree,
            smoothing_sigma=smoothing_sigma,
            max_frame=max_frame,
            logger=logger
        )
        
    elif method == "lowpass_filter":
        cutoff_freq = config.get("cutoff_freq", 0.001)
        temporal_filter_order = config.get("temporal_filter_order", 2)
        smoothing_sigma = config.get("smoothing_sigma", 2.0)
        
        if logger:
            logger.info(f"Using lowpass filter (cutoff={cutoff_freq}, order={temporal_filter_order})")
        
        # Calculate mean intensity per frame
        mean_f = np.mean(data, axis=(1, 2))
        n_frames = data.shape[0]
        
        # Design the filter
        nyq = 0.5  # Nyquist frequency
        normal_cutoff = cutoff_freq / nyq
        b, a = signal.butter(temporal_filter_order, normal_cutoff, btype='low')
        
        # Apply the filter to the mean intensity trace
        filtered_trend = signal.filtfilt(b, a, mean_f)
        
        # Calculate global mean
        global_mean = float(np.mean(data))
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Avoid division by zero
            factor = global_mean / max(filtered_trend[i], 1e-6)
            corrected_data[i] = data[i] * factor
        
        # Apply optional Gaussian smoothing
        if smoothing_sigma > 0:
            if logger:
                logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
            
            for i in range(n_frames):
                corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    elif method == "background_removal":
        # This applies background removal instead of photobleaching correction
        bg_method = config.get("background_method", "uniform")
        window_size = config.get("window_size", 7)
        
        if logger:
            logger.info(f"Using background removal method: {bg_method}, window size: {window_size}")
        
        corrected_data = remove_background(data, bg_method, window_size, logger=logger)
        
    elif method == "denoise":
        # This applies denoising instead of photobleaching correction
        denoise_method = config.get("denoise_method", "gaussian")
        denoise_params = config.get("denoise_params", {})
        
        if logger:
            logger.info(f"Using denoising method: {denoise_method}")
        
        corrected_data = denoise(data, denoise_method, logger=logger, **denoise_params)
        
    else:
        if logger:
            logger.warning(f"Photobleaching correction method '{method}' not recognized, returning original data")
        corrected_data = data.copy()
    
    # Calculate processing time
    processing_time = time.time() - start_time
    if logger:
        logger.info(f"Photobleaching correction completed in {processing_time:.2f} seconds")
    
    # Create verification plot if requested
    if config.get("generate_plot", False):
        plot_dir = os.path.dirname(output_path) if output_path else "."
        plot_name = f"correction_verification_{method}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        
        if logger:
            logger.info(f"Generating verification plot: {plot_path}")
            
        plot_photobleaching_correction(data, corrected_data, method, plot_path, logger)
    
    # Save results if output path is provided and save_output is True
    if output_path and save_output:
        if logger:
            logger.info(f"Saving corrected data to {output_path}")
            
        with h5py.File(output_path, 'w') as f:
            # Save corrected data
            f.create_dataset('corrected_data', data=corrected_data, compression='gzip', compression_opts=4)
            
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['method'] = method
            meta_group.attrs['processing_time'] = processing_time
            meta_group.create_dataset('image_shape', data=np.array(image_shape))
            
            # Save configuration
            config_group = f.create_group('config')
            for key, value in config.items():
                if isinstance(value, dict):
                    # Create subgroup for nested dictionaries
                    subgroup = config_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool, np.number)):
                            subgroup.attrs[subkey] = safe_hdf5_value(subvalue)
                elif isinstance(value, (int, float, str, bool, np.number)):
                    config_group.attrs[key] = safe_hdf5_value(value)
    elif output_path and not save_output:
        if logger:
            logger.info(f"Skipping saving corrected data to {output_path} as save_output=False")
    
    return corrected_data, image_shape


# Adapt for xarray input if xarray is available
if HAS_XARRAY:
    def correct_photobleaching_xarray(varr, method="polynomial_detrend", config=None, **kwargs):
        """
        Xarray-compatible wrapper for photobleaching correction.
        
        Parameters
        ----------
        varr : xarray.DataArray
            Input data with dimensions 'frame', 'height', 'width'
        method : str
            Correction method
        config : dict, optional
            Configuration parameters
        **kwargs
            Additional parameters
            
        Returns
        -------
        tuple
            (corrected_data, image_shape) - corrected xarray.DataArray and image shape
        """
        # Check if xarray is installed
        if not HAS_XARRAY:
            raise ImportError("xarray is required for this function")
        
        # Extract numpy array from xarray
        if 'frame' in varr.dims:
            data = varr.transpose('frame', 'height', 'width').values
        else:
            # Assume first dimension is frames if 'frame' not found
            data = varr.values
        
        # Get dimensions and create config
        if config is None:
            config = {"correction_method": method}
        else:
            config["correction_method"] = method
        
        # Add kwargs to config
        for key, value in kwargs.items():
            config[key] = value
        
        # Run correction on numpy array
        corrected_data, image_shape = correct_photobleaching(data, config=config)
        
        # Convert back to xarray
        if 'frame' in varr.dims:
            result = xr.DataArray(
                corrected_data,
                dims=varr.dims,
                coords=varr.coords,
                attrs=varr.attrs
            )
        else:
            # Create with original dimensions order
            result = xr.DataArray(
                corrected_data,
                dims=varr.dims,
                coords=varr.coords,
                attrs=varr.attrs
            )
        
        # Create a meaningful name suffix based on the method
        method_suffix = {
            "polynomial_detrend": "poly_detrend",
            "exponential_decay": "exp_decay",
            "bi_exponential": "bi_exp",
            "adaptive_percentile": "adapt_pct",
            "two_stage_detrend": "two_stage",
            "detrend_movie": "detrend",
            "lowpass_filter": "lowpass",
            "background_removal": "bg_removed",
            "denoise": "denoised"
        }.get(method, method)
        
        # Rename with method suffix
        result = result.rename(f"{varr.name}_bc_{method_suffix}" if varr.name else f"data_bc_{method_suffix}")
        
        return result, image_shape


def load_masks(masks_path, logger=None):
    """
    Load masks from various file formats for CNMF initialization.
    
    Parameters
    ----------
    masks_path : str
        Path to mask file (HDF5, NPY, or NumPy array)
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Array of masks with shape (n_masks, height, width)
    """
    import numpy as np
    import os
    
    if logger:
        logger.info(f"Loading masks from {masks_path}")
    
    file_ext = os.path.splitext(masks_path)[1].lower()
    
    if file_ext == '.h5' or file_ext == '.hdf5':
        # Load from HDF5
        import h5py
        with h5py.File(masks_path, 'r') as f:
            # Try common dataset names
            for name in ['masks', 'ROIs', 'rois', 'components']:
                if name in f:
                    masks = f[name][:]
                    if logger:
                        logger.info(f"Loaded masks from dataset '{name}'")
                    return masks.astype(np.float32)
            
            # If no standard name, take the first dataset
            for name in f.keys():
                if isinstance(f[name], h5py.Dataset):
                    masks = f[name][:]
                    if logger:
                        logger.info(f"Loaded masks from dataset '{name}'")
                    return masks.astype(np.float32)
    
    elif file_ext == '.npy':
        # Load from NPY
        masks = np.load(masks_path)
        if logger:
            logger.info(f"Loaded masks from NumPy file, shape: {masks.shape}")
        return masks.astype(np.float32)
    
    else:
        if logger:
            logger.warning(f"Unsupported mask file format: {file_ext}")
        return None


#################################
# NEW PREPROCESSING APPROACHES #
#################################

def remove_background_perframe(fm, method, wnd=None, selem=None):
    """
    Remove background from a single frame.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.
    method : str
        Method to use to remove background. Should be either "uniform" or "tophat".
    wnd : int, optional
        Size of the uniform filter. Only used if method == "uniform".
    selem : np.ndarray, optional
        Kernel used for morphological operations. Only used if method == "tophat".

    Returns
    -------
    fm : np.ndarray
        The frame with background removed.
    """
    if method == "uniform":
        if wnd is None:
            raise ValueError("Window size (wnd) must be provided for uniform method")
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        if selem is None:
            if not HAS_SKIMAGE:
                raise ImportError("skimage is required for tophat method")
            # Create a default disk element if not provided
            selem = disk(3)  # Default radius of 3
        return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)
    else:
        raise ValueError(f"Background removal method '{method}' not understood")


def remove_background(data, method, wnd, logger=None):
    """
    Remove background from a video.

    This function removes background frame by frame. Two methods are available:
    - "uniform": background is estimated by convolving with a uniform/mean kernel
    - "tophat": a morphological tophat operation is applied to each frame

    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    method : str
        The method used to remove the background. Either "uniform" or "tophat".
    wnd : int
        Window size for background removal, specified in pixels.
        If method == "uniform", this will be the size of a box kernel.
        If method == "tophat", this will be the radius of a disk kernel.
    logger : logging.Logger, optional
        Logger for status updates

    Returns
    -------
    numpy.ndarray
        The resulting movie with background removed.
    """
    if logger:
        logger.info(f"Removing background using {method} method (window size {wnd})")
    
    # Get dimensions
    n_frames, height, width = data.shape
    
    # Prepare kernel for tophat method
    selem = None
    if method == "tophat":
        if not HAS_SKIMAGE:
            raise ImportError("skimage.morphology.disk required for tophat method")
        selem = disk(wnd)
        
    # Process each frame
    result = np.zeros_like(data)
    for i in range(n_frames):
        result[i] = remove_background_perframe(data[i], method, wnd, selem)
        
        # Log progress periodically
        if logger and i % 100 == 0:
            logger.info(f"Processed {i}/{n_frames} frames")
    
    if logger:
        logger.info("Background removal completed")
    
    return result


def denoise_frame(frame, method, **kwargs):
    """
    Denoise a single frame using various methods.

    Parameters
    ----------
    frame : np.ndarray
        The input frame to denoise
    method : str
        Denoising method to use
    **kwargs
        Additional parameters for the specific denoising method

    Returns
    -------
    np.ndarray
        Denoised frame
    """
    if method == "gaussian":
        # Process ksize for GaussianBlur
        ksize = kwargs.get('ksize', (5, 5))
        sigma = kwargs.get('sigmaX', 0)
        return cv2.GaussianBlur(frame, ksize, sigma)
    
    elif method == "median":
        # Check if ksize is present and odd
        ksize = kwargs.get('ksize', 5)
        if isinstance(ksize, int) and ksize % 2 == 0:
            ksize += 1  # Make sure it's odd
        return cv2.medianBlur(frame, ksize)
    
    elif method == "bilateral":
        d = kwargs.get('d', 9)
        sigmaColor = kwargs.get('sigmaColor', 75)
        sigmaSpace = kwargs.get('sigmaSpace', 75)
        return cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
    
    elif method == "anisotropic":
        if not HAS_MEDPY:
            raise ImportError("medpy is required for anisotropic diffusion")
        
        niter = kwargs.get('niter', 5)
        kappa = kwargs.get('kappa', 50)
        gamma = kwargs.get('gamma', 0.1)
        option = kwargs.get('option', 1)
        return anisotropic_diffusion(frame, niter, kappa, gamma, option)
    
    else:
        raise ValueError(f"Denoising method '{method}' not understood")


def denoise(data, method, **kwargs):
    """
    Denoise the movie frame by frame.

    This function applies various denoising methods to each frame of the movie.

    Parameters
    ----------
    data : numpy.ndarray
        Input data with shape (frames, height, width)
    method : str
        The method to use to denoise each frame:
        - "gaussian": Gaussian filter using cv2.GaussianBlur
        - "anisotropic": Anisotropic filtering using medpy.filter.smoothing.anisotropic_diffusion
        - "median": Median filter using cv2.medianBlur
        - "bilateral": Bilateral filter using cv2.bilateralFilter
    **kwargs
        Additional parameters specific to each denoising method

    Returns
    -------
    numpy.ndarray
        The resulting denoised movie
    """
    logger = kwargs.pop('logger', None)
    if logger:
        logger.info(f"Applying {method} denoising")
    
    # Get dimensions
    n_frames, height, width = data.shape
    
    # Process each frame
    result = np.zeros_like(data)
    for i in range(n_frames):
        result[i] = denoise_frame(data[i], method, **kwargs)
        
        # Log progress periodically
        if logger and i % 100 == 0:
            logger.info(f"Processed {i}/{n_frames} frames")
    
    if logger:
        logger.info("Denoising completed")
    
    return result


def plot_photobleaching_correction(original_data, corrected_data, method="polynomial_detrend", save_path=None, logger=None):
    """
    Create a verification plot comparing original and corrected data.
    
    Parameters
    ----------
    original_data : numpy.ndarray
        The original movie data.
    corrected_data : numpy.ndarray
        The photobleaching-corrected movie data.
    method : str
        The correction method used.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed but not saved.
    logger : logging.Logger, optional
        Logger for status updates
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    
    if logger:
        logger.info(f"Generating photobleaching correction plot for method: {method}")
    
    # Calculate mean intensity per frame
    mean_original = np.mean(original_data, axis=(1, 2))
    mean_corrected = np.mean(corrected_data, axis=(1, 2))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Format method name for display
    method_display = method.replace('_', ' ').title()
    
    # Plot mean intensity over time
    ax1.plot(mean_original, 'r-', alpha=0.7, label='Original')
    ax1.plot(mean_corrected, 'g-', alpha=0.7, label=f'Corrected ({method_display})')
    ax1.set_title('Photobleaching Correction Verification', fontsize=14)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Mean Intensity')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot the correction factor (ratio between original and corrected)
    correction_factor = mean_corrected / np.maximum(mean_original, 1e-6)
    ax2.plot(correction_factor, 'b-', alpha=0.7)
    ax2.set_title('Correction Factor', fontsize=12)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Factor')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        if logger:
            logger.info(f"Saving photobleaching plot to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    
    return fig