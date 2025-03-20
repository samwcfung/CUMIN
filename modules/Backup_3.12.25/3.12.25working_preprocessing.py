#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Module
------------------
Implements photobleaching correction methods including polynomial detrending, CNMF,
CNMF-E for microendoscopic data, exponential fits, and low-pass filtering approaches.
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
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from scipy import signal

# Try to import CuPy for GPU acceleration, fallback to CPU if not available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not found, falling back to CPU processing for applicable methods")

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
    import numpy as np
    from skimage.transform import resize
    import time
    
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
    import numpy as np
    
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
        corrected_data[i] = data[i] * (global_mean / fitted_trend[i])
    
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
    from scipy.optimize import curve_fit
    
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
    from scipy.optimize import curve_fit
    
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
        stage1_data[i] = data[i] * (global_mean / fitted_trend1[i])
    
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
        corrected_data[i] = stage1_data[i] * (global_mean2 / fitted_trend2[i])
    
    if logger:
        logger.info(f"Completed second stage detrending with degree {second_degree}")
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data

def correct_photobleaching(varr: xr.DataArray, method: str = "polynomial_detrend", config=None, **kwargs) -> tuple:
    """
    Correct photobleaching in fluorescence imaging data.
    
    This function corrects photobleaching effects frame by frame by modeling the
    intensity decay over time and normalizing each frame.
    
    Parameters
    ----------
    varr : xr.DataArray
        The input movie data, should have dimensions "height", "width" and
        "frame".
    method : str
        The method used for photobleaching correction:
        - "polynomial_detrend": Fits a polynomial to mean intensities
        - "exponential_decay": Fits an exponential decay model
        - "bi_exponential": Fits a bi-exponential decay model
        - "adaptive_percentile": Uses a sliding window percentile approach
        - "two_stage_detrend": Applies two-stage polynomial detrending
        - "detrend_movie": Simple movie detrending
        - "lowpass_filter": Uses lowpass filter for correction
    config : dict, optional
        Configuration dictionary. If provided, parameters are read from this dictionary.
        Overrides individual kwargs if both are provided.
    **kwargs
        Additional parameters for the selected method:
        - polynomial_order (int): Degree of polynomial (for polynomial_detrend)
        - smoothing_sigma (float): Sigma for Gaussian smoothing
        - window_size (int): Size of sliding window (for adaptive_percentile)
        - percentile (float): Percentile value (for adaptive_percentile)
        - first_degree, second_degree (int): Polynomial degrees (for two_stage_detrend)
        - cutoff_freq (float): Cutoff frequency for lowpass filter
        - temporal_filter_order (int): Filter order for lowpass filtering
    
    Returns
    -------
    tuple
        (corrected_data, image_shape) - xarray.DataArray of corrected data and image shape tuple
    
    Notes
    -----
    For all methods, the approach is to:
    1. Calculate mean intensity per frame
    2. Fit a model to these values to estimate photobleaching trend
    3. Normalize each frame by the estimated trend
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import medfilt
    from scipy.ndimage import gaussian_filter
    
    # Extract parameters from config if provided, otherwise use kwargs or defaults
    if config is not None and isinstance(config, dict):
        # Get preprocessing section if it exists
        preproc_config = config.get('preprocessing', config)
        
        # Extract parameters with defaults
        polynomial_order = preproc_config.get('polynomial_order', kwargs.get('polynomial_order', 3))
        smoothing_sigma = preproc_config.get('smoothing_sigma', kwargs.get('smoothing_sigma', 2.0))
        first_degree = preproc_config.get('first_degree', kwargs.get('first_degree', 1))
        second_degree = preproc_config.get('second_degree', kwargs.get('second_degree', 3))
        window_size = preproc_config.get('window_size', kwargs.get('window_size', 101))
        percentile = preproc_config.get('percentile', kwargs.get('percentile', 10))
        cutoff_freq = preproc_config.get('cutoff_freq', kwargs.get('cutoff_freq', 0.001))
        temporal_filter_order = preproc_config.get('temporal_filter_order', kwargs.get('temporal_filter_order', 2))
        
        # Use method from config if not explicitly provided
        if method == "polynomial_detrend" and 'correction_method' in preproc_config:
            method = preproc_config['correction_method']
    else:
        # Use kwargs with defaults
        polynomial_order = kwargs.get('polynomial_order', 3)
        smoothing_sigma = kwargs.get('smoothing_sigma', 2.0)
        first_degree = kwargs.get('first_degree', 1)
        second_degree = kwargs.get('second_degree', 3)
        window_size = kwargs.get('window_size', 101)
        percentile = kwargs.get('percentile', 10)
        cutoff_freq = kwargs.get('cutoff_freq', 0.001)
        temporal_filter_order = kwargs.get('temporal_filter_order', 2)
    
    # Convert DataArray to numpy array for processing
    # Check dimensions of xarray to determine the frame dimension
    dims = varr.dims
    if 'frame' in dims:
        data = varr.transpose('frame', 'height', 'width').values
    else:
        # Assume first dimension is frames if 'frame' not found
        data = varr.values
    
    # Get dimensions and calculate mean intensity per frame
    n_frames = data.shape[0]
    mean_f = np.mean(data, axis=(1, 2))
    t = np.arange(n_frames)
    
    # Calculate global mean across all pixels and frames
    global_mean = float(np.mean(data))
    
    # Apply correction based on selected method
    if method == "polynomial_detrend":
        # Fit polynomial to mean fluorescence values
        poly_coeffs = np.polyfit(t, mean_f, polynomial_order)
        fitted_trend = np.polyval(poly_coeffs, t)
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Divide by trend and multiply by global mean (avoid division by zero)
            corrected_data[i] = data[i] * (global_mean / max(fitted_trend[i], 1e-6))
    
    elif method == "exponential_decay":
        # Define exponential decay function
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        # Initial parameter estimates
        p0 = [mean_f[0] - mean_f[-1], 1/n_frames, mean_f[-1]]
        
        try:
            # Fit the exponential decay model
            popt, _ = curve_fit(exp_decay, t, mean_f, p0=p0, maxfev=10000)
            fitted_trend = exp_decay(t, *popt)
        except:
            # Fall back to a simple exponential model if fitting fails
            b_est = -np.log(mean_f[-1]/mean_f[0]) / n_frames
            fitted_trend = mean_f[0] * np.exp(-b_est * t)
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Avoid division by zero
            factor = global_mean / max(fitted_trend[i], 1e-6)
            corrected_data[i] = data[i] * factor
    
    elif method == "bi_exponential":
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
        except:
            # Fall back to polynomial fit if bi-exponential fitting fails
            poly_coeffs = np.polyfit(t, mean_f, 3)
            fitted_trend = np.polyval(poly_coeffs, t)
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Avoid division by zero
            factor = global_mean / max(fitted_trend[i], 1e-6)
            corrected_data[i] = data[i] * factor
    
    elif method == "adaptive_percentile":
        window_size = kwargs.get('window_size', 101)
        percentile = kwargs.get('percentile', 10)
        
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
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Avoid division by zero
            factor = global_mean / max(trend[i], 1e-6)
            corrected_data[i] = data[i] * factor
    
    elif method == "two_stage_detrend":
        first_degree = kwargs.get('first_degree', 1)
        second_degree = kwargs.get('second_degree', 3)
        
        # STAGE 1: First pass with low-order polynomial
        poly_coeffs1 = np.polyfit(t, mean_f, first_degree)
        fitted_trend1 = np.polyval(poly_coeffs1, t)
        
        # Apply first correction
        stage1_data = np.zeros_like(data)
        for i in range(n_frames):
            stage1_data[i] = data[i] * (global_mean / max(fitted_trend1[i], 1e-6))
        
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
            factor = global_mean2 / max(fitted_trend2[i], 1e-6)
            corrected_data[i] = stage1_data[i] * factor
    
    elif method == "detrend_movie":
        # Simple detrending - similar to polynomial_detrend but with a different implementation
        detrend_degree = polynomial_order  # Use polynomial_order as detrend_degree
        
        # Fit polynomial to mean fluorescence values
        poly_coeffs = np.polyfit(t, mean_f, detrend_degree)
        fitted_trend = np.polyval(poly_coeffs, t)
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Divide by trend and multiply by global mean
            corrected_data[i] = data[i] * (global_mean / max(fitted_trend[i], 1e-6))
    
    elif method == "lowpass_filter":
        # Low-pass filter approach for photobleaching correction
        from scipy import signal
        
        # Design the filter
        nyq = 0.5  # Nyquist frequency
        normal_cutoff = cutoff_freq / nyq
        b, a = signal.butter(temporal_filter_order, normal_cutoff, btype='low')
        
        # Apply the filter to the mean intensity trace
        filtered_trend = signal.filtfilt(b, a, mean_f)
        
        # Apply correction
        corrected_data = np.zeros_like(data)
        for i in range(n_frames):
            # Avoid division by zero
            factor = global_mean / max(filtered_trend[i], 1e-6)
            corrected_data[i] = data[i] * factor
    
    else:
        raise NotImplementedError(f"Photobleaching correction method '{method}' not understood")
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    # Convert back to xarray DataArray
    # Ensure dimensions are correctly assigned
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
        "lowpass_filter": "lowpass"
    }.get(method, method)
    
    result = result.rename(varr.name + f"_bc_{method_suffix}")
    
    # Get image shape (height, width)
    if 'height' in varr.dims and 'width' in varr.dims:
        image_shape = (varr.sizes['height'], varr.sizes['width'])
    else:
        # If height/width dimensions aren't named as expected, use the last two dimensions
        image_shape = corrected_data.shape[-2:]
    
    return result, image_shape


def plot_photobleaching_correction(varr_original: xr.DataArray, varr_corrected: xr.DataArray, 
                                  method: str = "polynomial_detrend", save_path: str = None):
    """
    Create a verification plot comparing original and corrected data.
    
    Parameters
    ----------
    varr_original : xr.DataArray
        The original movie data.
    varr_corrected : xr.DataArray
        The photobleaching-corrected movie data.
    method : str
        The correction method used.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed but not saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    
    # Calculate mean intensity per frame
    # Handle different dimension names
    spatial_dims = [dim for dim in varr_original.dims if dim in ['height', 'width']]
    mean_original = varr_original.mean(dim=spatial_dims).values
    mean_corrected = varr_corrected.mean(dim=spatial_dims).values
    
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
    correction_factor = mean_corrected / mean_original
    ax2.plot(correction_factor, 'b-', alpha=0.7)
    ax2.set_title('Correction Factor', fontsize=12)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Factor')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    
    return fig

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