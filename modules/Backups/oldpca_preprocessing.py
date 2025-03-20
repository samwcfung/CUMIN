#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module
------------------
Implements photobleaching correction methods including PCA-based correction,
bi-exponential fit, and binomial exponential correction.
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
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# Try to import CuPy for GPU acceleration, fallback to CPU if not available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not found, falling back to CPU processing for applicable methods")

def correct_photobleaching(
    tif_path, 
    output_path, 
    config, 
    logger
):
    """
    Correct photobleaching using the specified method.
    
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
    n_components = config.get("n_components", 2)  # For PCA method
    
    logger.info(f"Using correction method: {correction_method}")
    if correction_method in ["binomial_exponential", "exponential_fit"]:
        logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Apply photobleaching correction based on method
    if correction_method == "pca":
        # Use PCA for photobleaching correction
        corrected_data = pca_correction(
            data,
            n_components=n_components,
            logger=logger
        )
    elif correction_method == "bi_exponential_fit":
        # Use bi-exponential fit method
        corrected_data = bi_exponential_fit_correction(
            data, 
            smoothing_sigma=smoothing_sigma,
            use_gpu=use_gpu,
            logger=logger
        )
    elif correction_method == "binomial_exponential":
        # Use binomial exponential correction
        corrected_data = binomial_exponential_correction(
            data, 
            binomial_order=binomial_order, 
            smoothing_sigma=smoothing_sigma,
            use_gpu=use_gpu,
            logger=logger
        )
    elif correction_method == "exponential_fit":
        # Use exponential fit method
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
        if correction_method == "binomial_exponential":
            meta_group.attrs['binomial_order'] = binomial_order
        if correction_method == "pca":
            meta_group.attrs['n_components'] = n_components
        meta_group.attrs['smoothing_sigma'] = smoothing_sigma
        meta_group.attrs['gpu_accelerated'] = use_gpu and correction_method != "pca"
        meta_group.attrs['source_file'] = Path(tif_path).name
    
    # Create a verification plot
    try:
        # Create verification plot
        plt.figure(figsize=(12, 6))
        
        # Plot mean intensity over time
        plt.plot(np.mean(data, axis=(1, 2)), 'r-', alpha=0.7, label='Original')
        plt.plot(np.mean(corrected_data, axis=(1, 2)), 'g-', alpha=0.7, label=f'{correction_method.replace("_", " ").title()} Corrected')
        
        plt.title('Photobleaching Correction Verification')
        plt.xlabel('Frame')
        plt.ylabel('Mean Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        verification_path = os.path.join(os.path.dirname(output_path), f"{Path(tif_path).stem}_bleach_correction.png")
        plt.savefig(verification_path, dpi=150)
        plt.close()
        logger.info(f"Saved correction verification plot to {verification_path}")
    except Exception as e:
        logger.warning(f"Could not create verification plot: {str(e)}")
    
    logger.info(f"Photobleaching correction completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Saved corrected data to {output_path}")
    
    return corrected_data, image_shape

def pca_correction(data, n_components=2, logger=None):
    """
    Correct photobleaching using PCA-based correction.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    n_components : int
        Number of PCA components to consider as photobleaching (default=2)
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Corrected fluorescence data
    """
    try:
        from sklearn.decomposition import PCA
        
        if logger:
            logger.info(f"Starting PCA-based photobleaching correction with {n_components} components")
        
        # Get data dimensions
        n_frames, height, width = data.shape
        n_pixels = height * width
        
        # Reshape data for PCA: (n_frames, n_pixels)
        X = data.reshape(n_frames, -1)
        
        # Ensure n_components doesn't exceed the number of frames
        n_components = min(n_components, n_frames - 1)
        
        # Run PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        
        if logger:
            explained_var = pca.explained_variance_ratio_
            logger.info(f"Top {n_components} PCA components explain {np.sum(explained_var):.2%} of variance")
            logger.info(f"Individual components: {[f'{var:.2%}' for var in explained_var]}")
        
        # Reconstruct the photobleaching component
        bleaching_component = pca.inverse_transform(components)
        
        # Subtract just the first component (typically photobleaching)
        # Keep the rest of the components (neuronal activity)
        corrected_flat = X.copy()
        
        # For each component, check if it looks like a photobleaching trend
        selected_components = []
        
        for i in range(n_components):
            comp = components[:, i]
            
            # Check if this component has a consistent trend like photobleaching
            # 1. Compute correlations with a linear trend
            t = np.arange(len(comp)) / len(comp)
            lin_corr = np.corrcoef(comp, t)[0, 1]
            
            # 2. See if component is largely monotonic
            increases = np.sum(np.diff(comp) > 0)
            decreases = np.sum(np.diff(comp) < 0)
            monotonicity = max(increases, decreases) / (increases + decreases) if (increases + decreases) > 0 else 0
            
            # If it looks like photobleaching, mark it for removal
            if abs(lin_corr) > 0.7 or monotonicity > 0.8:
                selected_components.append(i)
                if logger:
                    logger.info(f"Component {i+1} selected as photobleaching (corr: {lin_corr:.2f}, monotonicity: {monotonicity:.2f})")
        
        # If we didn't find components that look like photobleaching, just use the top component
        if not selected_components:
            selected_components = [0]
            if logger:
                logger.info("No clear photobleaching components found. Using first component as fallback.")
        
        # Create a mask for selected components
        mask = np.zeros(n_components, dtype=bool)
        mask[selected_components] = True
        
        # Create the photobleaching pattern by extracting only selected components
        components_subset = np.zeros_like(components)
        components_subset[:, mask] = components[:, mask]
        bleaching_pattern = pca.inverse_transform(components_subset)
        
        # Subtract the photobleaching pattern
        corrected_flat = X - bleaching_pattern
        
        # Calculate a scaling factor to maintain original mean intensity
        original_mean = np.mean(X, axis=0, keepdims=True)
        corrected_mean = np.mean(corrected_flat, axis=0, keepdims=True)
        
        # Replace zeros to avoid division errors
        corrected_mean[corrected_mean == 0] = 1
        
        # Apply scaling
        corrected_flat = corrected_flat * (original_mean / corrected_mean)
        
        # Ensure data remains positive
        min_val = np.min(corrected_flat)
        if min_val < 0:
            corrected_flat -= min_val
        
        # Reshape back to original dimensions
        corrected_data = corrected_flat.reshape(n_frames, height, width)
        
        # Ensure data type matches original
        corrected_data = corrected_data.astype(data.dtype)
        
        if logger:
            logger.info("PCA correction completed successfully")
        
        return corrected_data
        
    except ImportError:
        if logger:
            logger.warning("scikit-learn not installed, falling back to bi-exponential method")
        return bi_exponential_fit_correction(data, logger=logger)
    except Exception as e:
        if logger:
            logger.warning(f"PCA correction failed: {str(e)}, falling back to bi-exponential method")
        return bi_exponential_fit_correction(data, logger=logger)

def bi_exponential_fit_correction(data, smoothing_sigma=2.0, use_gpu=True, logger=None):
    """
    Correct photobleaching using bi-exponential fit method.
    
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
    if logger:
        logger.info("Applying bi-exponential fit correction")
    
    # Define bi-exponential decay function
    def bi_exp(t, a, b, c, d):
        return (a * np.exp(-b * t)) + (c * np.exp(-d * t))
    
    # Calculate mean fluorescence per frame
    if use_gpu and HAS_CUPY:
        data_gpu = cp.asarray(data)
        mean_f = cp.asnumpy(cp.mean(data_gpu, axis=(1, 2)))
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        mean_f = np.mean(data, axis=(1, 2))
    
    # Timepoints (frame indices)
    t = np.arange(len(mean_f))
    
    # Fit bi-exponential decay
    try:
        # Initial parameter guess
        p0 = [
            mean_f[0] * 0.6,  # a: first component amplitude
            0.05,             # b: first decay rate
            mean_f[0] * 0.4,  # c: second component amplitude
            0.005             # d: second decay rate
        ]
        
        # Define bounds
        lower_bounds = [0, 0, 0, 0]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        
        if logger:
            logger.info("Fitting bi-exponential decay model")
        
        # Fit curve
        popt, _ = curve_fit(bi_exp, t, mean_f, p0=p0, bounds=(lower_bounds, upper_bounds))
        
        # Calculate fitted curve
        fitted = bi_exp(t, *popt)
        
        # Calculate correction factors
        f = fitted / np.max(fitted)  # Normalize
        correction_factor = 1.0 / f
        
        # Make sure correction_factor has the right length
        if len(correction_factor) < data.shape[0]:
            # Extend correction factor if needed
            padding = np.ones(data.shape[0] - len(correction_factor)) * correction_factor[-1]
            correction_factor = np.concatenate([correction_factor, padding])
        elif len(correction_factor) > data.shape[0]:
            # Truncate if too long
            correction_factor = correction_factor[:data.shape[0]]
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            corrected_data[i] = data[i] * correction_factor[i]
        
    except (RuntimeError, ValueError) as e:
        if logger:
            logger.warning(f"Bi-exponential fit failed: {str(e)}, falling back to exponential fit")
        return exponential_fit_correction(data, smoothing_sigma=smoothing_sigma, use_gpu=use_gpu, logger=logger)
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        # Apply smoothing to each frame
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data

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
    
    if use_gpu and HAS_CUPY:
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
    if logger:
        logger.info("Applying exponential fit correction")
    
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
    
    # Timepoints (frame indices)
    t = np.arange(len(mean_f))
    
    # Fit exponential decay
    try:
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
        corrected_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            corrected_data[i] = data[i] * correction_factor[i]
        
    except (RuntimeError, ValueError) as e:
        if logger:
            logger.warning(f"Exponential fit failed: {str(e)}, falling back to binomial filter")
        return binomial_exponential_correction(data, smoothing_sigma=smoothing_sigma, use_gpu=use_gpu, logger=logger)
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        # Apply smoothing to each frame
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data