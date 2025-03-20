#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module
------------------
Implements photobleaching correction methods including polynomial detrending and CNMF.
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
    logger.info(f"==== DEBUG ====")
    logger.info(f"Preprocessing config received: {config}")
    logger.info(f"Correction method: {config.get('correction_method', 'NOT_SPECIFIED')}")
    logger.info(f"==== END DEBUG ====")
    
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
    correction_method = config.get("correction_method", "polynomial_detrend")
    polynomial_order = config.get("polynomial_order", 3)
    smoothing_sigma = config.get("smoothing_sigma", 2.0)
    use_gpu = config.get("use_gpu", True) and HAS_CUPY
    num_components = config.get("num_components", None)
    apply_cnmf = config.get("apply_cnmf", True)
    
    logger.info(f"Using correction method: {correction_method}")
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Apply photobleaching correction based on method
    # Inside correct_photobleaching function, update the method selection
    if correction_method == "exponential_detrend":
        # Use exponential detrending method
        corrected_data = exponential_detrend_correction(
            data,
            smoothing_sigma=smoothing_sigma,
            logger=logger
        )
    elif correction_method == "polynomial_detrend":
        # Use polynomial detrending method
        corrected_data = polynomial_detrend_correction(
            data,
            poly_order=polynomial_order,
            smoothing_sigma=smoothing_sigma,
            use_gpu=use_gpu,
            logger=logger
        )
    elif correction_method == "cnmf":
        # Use CNMF for photobleaching correction
        corrected_data = cnmf_correction(
            data,
            num_components=num_components,
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
        binomial_order = config.get("binomial_order", 3)
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
    
    # Apply CNMF for signal refinement if requested and not already using CNMF
    if apply_cnmf and correction_method != "cnmf":
        logger.info("Applying CNMF for signal refinement")
        try:
            refined_data = cnmf_correction(
                corrected_data,
                num_components=num_components,
                logger=logger
            )
            logger.info("CNMF refinement completed successfully")
            final_data = refined_data
        except Exception as e:
            logger.warning(f"CNMF refinement failed: {str(e)}, using corrected data without CNMF")
            final_data = corrected_data
    else:
        final_data = corrected_data
    
    # Save corrected data to HDF5
    with h5py.File(output_path, 'w') as f:
        # Create dataset for corrected data
        f.create_dataset('corrected_data', data=final_data)
        
        # Save metadata
        meta_group = f.create_group('metadata')
        meta_group.create_dataset('image_shape', data=np.array(image_shape))
        meta_group.attrs['n_frames'] = n_frames
        meta_group.attrs['correction_method'] = correction_method
        if correction_method == "polynomial_detrend":
            meta_group.attrs['polynomial_order'] = polynomial_order
        if correction_method == "binomial_exponential":
            meta_group.attrs['binomial_order'] = config.get("binomial_order", 3)
        meta_group.attrs['apply_cnmf'] = apply_cnmf
        if apply_cnmf or correction_method == "cnmf":
            meta_group.attrs['num_components'] = num_components
        meta_group.attrs['smoothing_sigma'] = smoothing_sigma
        meta_group.attrs['gpu_accelerated'] = use_gpu and correction_method != "cnmf"
        meta_group.attrs['source_file'] = Path(tif_path).name
    
    # Create a verification plot
    try:
        # Create verification plot
        plt.figure(figsize=(12, 6))
        
        # Plot mean intensity over time
        plt.plot(np.mean(data, axis=(1, 2)), 'r-', alpha=0.7, label='Original')
        plt.plot(np.mean(final_data, axis=(1, 2)), 'g-', alpha=0.7, label=f'{correction_method.replace("_", " ").title()}' + 
                (' + CNMF' if apply_cnmf and correction_method != "cnmf" else ''))
        
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
    
    return final_data, image_shape

def polynomial_detrend_correction(data, poly_order=3, smoothing_sigma=2.0, use_gpu=False, logger=None):
    """
    Correct photobleaching using polynomial detrending method precisely following the paper description.
    """
    if logger:
        logger.info(f"Applying polynomial detrending (order={poly_order})")
    
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    
    # Timepoints (frame indices)
    t = np.arange(data.shape[0])
    
    # Fit polynomial to mean fluorescence values
    poly_coeffs = np.polyfit(t, mean_f, poly_order)
    fitted_trend = np.polyval(poly_coeffs, t)
    
    # Calculate global mean across all pixels and frames
    global_mean = float(np.mean(data))
    
    if logger:
        logger.info(f"Fitted {poly_order}-order polynomial for detrending")
        logger.info(f"Global mean intensity: {global_mean:.4f}")
    
    # Apply correction EXACTLY as stated in the paper:
    # "We then subtracted all pixels values in a given frame from the fitted values at each time point"
    # This means: (fitted_value - pixel_values) + global_mean
    corrected_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        # Note the order of subtraction here - this is key
        corrected_data[i] = fitted_trend[i] - data[i] + global_mean
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data

def cnmf_correction(data, num_components=None, logger=None):
    """
    Correct photobleaching using CNMF approach from CaImAn.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    num_components : int, optional
        Number of neural components to extract
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Corrected fluorescence data
    """
    try:
        import caiman as cm
        from caiman.source_extraction import cnmf
        
        if logger:
            logger.info("Starting CNMF-based photobleaching correction")
            logger.info(f"Using CaImAn version: {cm.__version__}")
        
        # Determine number of components if not specified
        if num_components is None:
            num_components = min(30, data.shape[0]//3)  # Reasonable default
            if logger:
                logger.info(f"Automatically set number of components to {num_components}")
        
        # For CaImAn 1.11.3, use the correct parameter naming
        params_dict = {
            'dims': data.shape[1:],          # Image dimensions
            'method_init': 'greedy_roi',     # Initialization method
            'K': num_components,             # Number of components (capital K in v1.11)
            'gSig': [4, 4],                  # Expected size of neurons (pixels)
            'p': 1,                          # Order of AR model for calcium dynamics
            'gnb': 2,                        # Number of background components (changed from nb to gnb)
            'merge_thresh': 0.8,             # Merging threshold (changed from merge_thr to merge_thresh)
        }
        
        # Initialize parameters for CNMF
        params = cnmf.params.CNMFParams(params_dict=params_dict)
        
        # Run CNMF
        if logger:
            logger.info("Running CNMF algorithm")
        
        # Convert data to CaImAn format (float32)
        Y = data.astype(np.float32)
        
        # Initialize CNMF
        cnm = cnmf.CNMF(n_processes=1, params=params, dview=None)
        
        # Fit CNMF model
        cnm = cnm.fit(Y)
        
        # Extract components
        if logger:
            logger.info("Extracting corrected components")
        
        # Get background components (contains photobleaching)
        background = cnm.b.dot(cnm.f)
        
        # Get neural components (denoised signals)
        neural = cnm.A.dot(cnm.C)
        
        # Determine baseline level (to keep overall brightness)
        baseline_level = np.median(background.reshape(data.shape[0], -1), axis=1)
        
        # Reconstruct corrected data
        corrected_data = np.zeros_like(data, dtype=np.float32)
        for t in range(data.shape[0]):
            # Add neural signals to baseline
            frame_neural = neural[:, t].reshape(data.shape[1:])
            corrected_data[t] = frame_neural + baseline_level[t]
        
        # Scale back to original range
        orig_min, orig_max = np.min(data), np.max(data)
        corr_min, corr_max = np.min(corrected_data), np.max(corrected_data)
        
        # Rescale if needed
        if corr_max > corr_min:
            corrected_data = (corrected_data - corr_min) / (corr_max - corr_min) * (orig_max - orig_min) + orig_min
        
        if logger:
            logger.info("CNMF correction completed successfully")
        
        return corrected_data
        
    except ImportError:
        if logger:
            logger.warning("CaImAn not installed, falling back to polynomial detrend method")
        return polynomial_detrend_correction(data, logger=logger)
    except Exception as e:
        if logger:
            logger.warning(f"CNMF correction failed: {str(e)}, falling back to polynomial detrend method")
        return polynomial_detrend_correction(data, logger=logger)

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
            logger.warning(f"Bi-exponential fit failed: {str(e)}, falling back to polynomial detrend")
        return polynomial_detrend_correction(data, poly_order=3, smoothing_sigma=smoothing_sigma, use_gpu=use_gpu, logger=logger)
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        # Apply smoothing to each frame
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data

def exponential_detrend_correction(data, smoothing_sigma=2.0, logger=None):
    """
    Correct photobleaching using an exponential model, which better represents
    the physics of photobleaching.
    """
    if logger:
        logger.info(f"Applying exponential detrending")
    
    # Calculate mean fluorescence per frame
    mean_f = np.mean(data, axis=(1, 2))
    
    # Timepoints (frame indices)
    t = np.arange(data.shape[0])
    
    # Fit exponential decay: f(t) = a * exp(-b * t) + c
    from scipy.optimize import curve_fit
    
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    try:
        # Initial parameter guess
        p0 = [mean_f[0] - mean_f[-1], 1.0/len(t), mean_f[-1]]
        
        # Fit curve with bounds to ensure stability
        popt, _ = curve_fit(exp_decay, t, mean_f, p0=p0, 
                           bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        
        # Calculate fitted curve
        fitted_trend = exp_decay(t, *popt)
        
        # Use first frame as reference
        reference_level = fitted_trend[0]
        
        if logger:
            logger.info(f"Fitted exponential decay: a={popt[0]:.4f}, b={popt[1]:.6f}, c={popt[2]:.4f}")
            logger.info(f"Reference level: {reference_level:.4f}")
        
        # Apply multiplicative correction
        corrected_data = np.zeros_like(data, dtype=np.float32)
        
        for i in range(data.shape[0]):
            if fitted_trend[i] > 0:
                # Scale each pixel by the inverse of the bleaching trend
                corrected_data[i] = data[i] * (reference_level / fitted_trend[i])
            else:
                corrected_data[i] = data[i]
                
        # Apply optional Gaussian smoothing
        if smoothing_sigma > 0:
            if logger:
                logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
            
            for i in range(corrected_data.shape[0]):
                corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
        
        return corrected_data
        
    except (RuntimeError, ValueError) as e:
        if logger:
            logger.warning(f"Exponential fit failed: {str(e)}, falling back to polynomial method")
        
        # Fall back to polynomial method with higher order
        return polynomial_detrend_correction(data, poly_order=3, 
                                           smoothing_sigma=smoothing_sigma, 
                                           use_gpu=False, logger=logger)

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
            logger.warning(f"Exponential fit failed: {str(e)}, falling back to polynomial detrend")
        return polynomial_detrend_correction(data, poly_order=3, smoothing_sigma=smoothing_sigma, use_gpu=use_gpu, logger=logger)
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        # Apply smoothing to each frame
        for i in range(corrected_data.shape[0]):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data