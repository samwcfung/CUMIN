#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module
------------------
Implements photobleaching correction and other preprocessing steps.
Utilizes GPU acceleration via CuPy when available.
"""

import os
import time
import numpy as np
import h5py
import tifffile
import warnings
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Try to import CuPy for GPU acceleration, fallback to CPU if not available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not found, falling back to CPU processing for photobleaching correction")

def correct_photobleaching(
    tif_path, 
    output_path, 
    config, 
    logger
):
    """
    Correct photobleaching using binomial exponential correction.
    
    Parameters
    ----------
    tif_path : str
        Path to the .tif video file
    output_path : str
        Path to save the corrected data (HDF5)
    config : dict
        Configuration parameters
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    tuple
        (corrected_data, image_shape) - NumPy array of corrected data and image shape
    """
    start_time = time.time()
    logger.info(f"Starting photobleaching correction for {Path(tif_path).name}")
    
    # Load .tif file and get metadata
    with tifffile.TiffFile(tif_path) as tif:
        # Get image shape
        if len(tif.pages) > 0:
            image_shape = (tif.pages[0].shape[0], tif.pages[0].shape[1])
            logger.info(f"Detected image shape: {image_shape}")
        else:
            raise ValueError(f"Could not determine image shape from {tif_path}")
        
        # Read all frames
        logger.info(f"Reading all frames from {tif_path}")
        data = tif.asarray()
        
    n_frames = data.shape[0]
    logger.info(f"Loaded {n_frames} frames with shape {data.shape}")
    
    # Extract correction parameters
    correction_method = config.get("correction_method", "binomial_exponential")
    binomial_order = config.get("binomial_order", 3)
    smoothing_sigma = config.get("smoothing_sigma", 2.0)
    use_gpu = config.get("use_gpu", True) and HAS_CUPY
    
    logger.info(f"Using correction method: {correction_method}")
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Apply photobleaching correction based on method
    if correction_method == "binomial_exponential":
        corrected_data = binomial_exponential_correction(
            data, 
            binomial_order=binomial_order, 
            smoothing_sigma=smoothing_sigma,
            use_gpu=use_gpu,
            logger=logger
        )
    elif correction_method == "exponential_fit":
        corrected_data = exponential_fit_correction(
            data, 
            smoothing_sigma=smoothing_sigma,
            use_gpu=use_gpu,
            logger=logger
        )
    else:
        raise ValueError(f"Unknown correction method: {correction_method}")
    
    # Save corrected data to HDF5
    with h5py.File(output_path, 'w') as f:
        # Create dataset for corrected data
        f.create_dataset('corrected_data', data=corrected_data)
        
        # Save metadata
        meta_group = f.create_group('metadata')
        meta_group.create_dataset('image_shape', data=np.array(image_shape))
        meta_group.attrs['n_frames'] = n_frames
        meta_group.attrs['correction_method'] = correction_method
        meta_group.attrs['binomial_order'] = binomial_order
        meta_group.attrs['smoothing_sigma'] = smoothing_sigma
        meta_group.attrs['gpu_accelerated'] = use_gpu
        meta_group.attrs['source_file'] = Path(tif_path).name
        
    logger.info(f"Photobleaching correction completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Saved corrected data to {output_path}")
    
    return corrected_data, image_shape

def binomial_exponential_correction(data, binomial_order=3, smoothing_sigma=2.0, use_gpu=True, logger=None):
    """
    Correct photobleaching using binomial exponential correction.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    binomial_order : int
        Order of binomial filter
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    use_gpu : bool
        Whether to use GPU acceleration
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Corrected fluorescence data
    """
    if logger:
        logger.info(f"Applying binomial exponential correction (order={binomial_order})")
    
    # Timepoints (frame indices)
    t = np.arange(data.shape[0])
    
    if use_gpu:
        # Transfer data to GPU
        if logger:
            logger.info("Transferring data to GPU")
        data_gpu = cp.asarray(data)
        
        # Calculate mean fluorescence per frame
        mean_f = cp.mean(data_gpu, axis=(1, 2))
        
        # Apply binomial filter to get baseline trend
        if logger:
            logger.info(f"Computing binomial filter (order={binomial_order})")
        
        # Generate binomial coefficients
        n = binomial_order
        k = cp.arange(n + 1)
        binomial_coef = cp.array([cp.math.comb(n, int(ki)) for ki in k])
        binomial_coef = binomial_coef / cp.sum(binomial_coef)
        
        # Extend the mean_f array with edge values for convolution
        pad_width = n // 2
        mean_f_padded = cp.pad(mean_f, pad_width, mode='edge')
        
        # Apply binomial filter
        filtered_f = cp.convolve(mean_f_padded, binomial_coef, mode='valid')
        
        # Calculate correction factor
        first_val = filtered_f[0]
        correction_factor = first_val / filtered_f
        
        # Make sure correction_factor has the right length
        if len(correction_factor) < data.shape[0]:
            # Extend correction factor if needed
            padding = cp.ones(data.shape[0] - len(correction_factor)) * correction_factor[-1]
            correction_factor = cp.concatenate([correction_factor, padding])
        elif len(correction_factor) > data.shape[0]:
            # Truncate if too long
            correction_factor = correction_factor[:data.shape[0]]
        
        # Apply correction to each frame
        if logger:
            logger.info("Applying correction factors")
        
        # Reshape correction factor for broadcasting
        correction_factor = correction_factor.reshape(-1, 1, 1)
        
        # Apply correction
        corrected_data_gpu = data_gpu * correction_factor
        
        # Transfer result back to CPU
        corrected_data = cp.asnumpy(corrected_data_gpu)
        
        # Clear GPU memory
        del data_gpu, corrected_data_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
    else:
        # CPU implementation
        # Calculate mean fluorescence per frame
        mean_f = np.mean(data, axis=(1, 2))
        
        # Apply binomial filter to get baseline trend
        if logger:
            logger.info(f"Computing binomial filter (order={binomial_order})")
        
        # Generate binomial coefficients
        n = binomial_order
        k = np.arange(n + 1)
        binomial_coef = np.array([np.math.comb(n, int(ki)) for ki in k])
        binomial_coef = binomial_coef / np.sum(binomial_coef)
        
        # Extend the mean_f array with edge values for convolution
        pad_width = n // 2
        mean_f_padded = np.pad(mean_f, pad_width, mode='edge')
        
        # Apply binomial filter
        filtered_f = np.convolve(mean_f_padded, binomial_coef, mode='valid')
        
        # Calculate correction factor
        first_val = filtered_f[0]
        correction_factor = first_val / filtered_f
        
        # Make sure correction_factor has the right length
        if len(correction_factor) < data.shape[0]:
            # Extend correction factor if needed
            padding = np.ones(data.shape[0] - len(correction_factor)) * correction_factor[-1]
            correction_factor = np.concatenate([correction_factor, padding])
        elif len(correction_factor) > data.shape[0]:
            # Truncate if too long
            correction_factor = correction_factor[:data.shape[0]]
        
        # Apply correction to each frame
        if logger:
            logger.info("Applying correction factors")
        
        # Apply correction using broadcasting
        corrected_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            corrected_data[i] = data[i] * correction_factor[i]
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        # Apply smoothing to each frame
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data

def exponential_fit_correction(data, smoothing_sigma=2.0, use_gpu=True, logger=None):
    """
    Correct photobleaching using exponential fit method.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    smoothing_sigma : float
        Sigma for Gaussian smoothing
    use_gpu : bool
        Whether to use GPU acceleration
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Corrected fluorescence data
    """
    from scipy.optimize import curve_fit
    
    if logger:
        logger.info("Applying exponential fit correction")
    
    # Timepoints (frame indices)
    t = np.arange(data.shape[0])
    
    # Define exponential decay function
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    # Calculate mean fluorescence per frame
    if use_gpu and HAS_CUPY:
        data_gpu = cp.asarray(data)
        mean_f = cp.asnumpy(cp.mean(data_gpu, axis=(1, 2)))
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        mean_f = np.mean(data, axis=(1, 2))
    
    # Fit exponential decay to mean fluorescence
    try:
        if logger:
            logger.info("Fitting exponential decay model")
        
        # Initial parameter guess
        p0 = [mean_f[0] - mean_f[-1], 1.0/len(t), mean_f[-1]]
        
        # Fit curve
        popt, _ = curve_fit(exp_decay, t, mean_f, p0=p0, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        
        # Calculate fitted curve
        fitted = exp_decay(t, *popt)
        
        # Calculate correction factors
        correction_factor = mean_f[0] / fitted
        
        # Make sure correction_factor has the right length
        if len(correction_factor) < data.shape[0]:
            # Extend correction factor if needed
            padding = np.ones(data.shape[0] - len(correction_factor)) * correction_factor[-1]
            correction_factor = np.concatenate([correction_factor, padding])
        elif len(correction_factor) > data.shape[0]:
            # Truncate if too long
            correction_factor = correction_factor[:data.shape[0]]
        
        # Apply correction
        if logger:
            logger.info("Applying correction factors")
        
        corrected_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            corrected_data[i] = data[i] * correction_factor[i]
        
    except RuntimeError:
        if logger:
            logger.warning("Exponential fit failed, falling back to binomial filter")
        return binomial_exponential_correction(data, smoothing_sigma=smoothing_sigma, use_gpu=use_gpu, logger=logger)
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        # Apply smoothing to each frame
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data