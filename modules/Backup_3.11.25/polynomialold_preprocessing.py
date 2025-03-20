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
        # Divide by trend and multiply by global mean to keep values in similar range
        # This prevents negative values while properly correcting for photobleaching
        corrected_data[i] = data[i] * (global_mean / fitted_trend[i])
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    return corrected_data

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
    if correction_method == "detrend_movie":
        # Use the new detrend_movie method
        detrend_degree = config.get("polynomial_order", 1)
        max_frame = config.get("max_frame", None)
        logger.info(f"Applying movie detrending with degree {detrend_degree}")
        corrected_data = detrend_movie(
            data,
            detrend_degree=detrend_degree,
            smoothing_sigma=smoothing_sigma,
            max_frame=max_frame,
            logger=logger
        )
    elif correction_method == "lowpass_filter":
        # Use low-pass filter correction (new method)
        cutoff_freq = config.get("cutoff_freq", 0.001)
        logger.info(f"Applying lowpass filter with cutoff_freq={cutoff_freq}")
        corrected_data = lowpass_filter_correction(
            data,
            cutoff_freq=cutoff_freq,
            smoothing_sigma=smoothing_sigma,
            memory_efficient=True,  # Enable memory-efficient processing
            max_chunk_size=50,      # Process in smaller chunks
            logger=logger
        )
    elif correction_method == "exponential_detrend":
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
    elif correction_method == "cnmf_e":
        # Use CNMF-E for microendoscopic data
        # Check if masks are provided
        masks_path = config.get("masks_path", None)
        masks = None
        
        if masks_path and os.path.exists(masks_path):
            try:
                # Load masks
                masks = load_masks(masks_path, logger)
                if masks is not None and masks.shape[0] > 0:
                    logger.info(f"Loaded {masks.shape[0]} custom masks for CNMF-E initialization from {masks_path}")
            except Exception as e:
                logger.warning(f"Could not load masks from {masks_path}: {str(e)}")
        
        # Extract CNMF-E parameters from config
        cnmf_e_params = config.get("cnmf_e", {})
        
        # Use CNMF-E for photobleaching correction
        try:
            corrected_data = cnmf_e_correction(
                data,
                masks=masks,
                num_components=num_components,
                min_corr=cnmf_e_params.get("min_corr", 0.8),
                min_pnr=cnmf_e_params.get("min_pnr", 10),
                gSig=cnmf_e_params.get("gSig", (4, 4)),
                gSiz=cnmf_e_params.get("gSiz", (15, 15)),
                ring_size_factor=cnmf_e_params.get("ring_size_factor", 1.5),
                merge_thresh=cnmf_e_params.get("merge_thresh", 0.8),
                use_multiprocessing=config.get("use_multiprocessing", False),
                n_processes=config.get("n_processes", 1),
                logger=logger
            )
        except Exception as e:
            logger.error(f"CNMF-E failed with error: {str(e)}")
            logger.info("Falling back to lowpass filter for photobleaching correction")
            
            # Fall back to lowpass filter
            corrected_data = lowpass_filter_correction(
                data,
                cutoff_freq=config.get("cutoff_freq", 0.001),
                smoothing_sigma=smoothing_sigma,
                logger=logger
            )
        
        # Store components in the output file
        if 'cnmf_e_components' in globals():
            # Store reference to components for future use
            global_components = globals()['cnmf_e_components']
            
            # Create extra output to store components if requested
            components_path = os.path.join(os.path.dirname(output_path), f"{Path(tif_path).stem}_cnmf_components.h5")
            try:
                logger.info(f"Saving CNMF-E components to {components_path}")
                with h5py.File(components_path, 'w') as f:
                    # Save components in a useful format
                    comp_group = f.create_group('cnmf_components')
                    
                    # Save spatial components (A) - convert from sparse if needed
                    if global_components.get('A') is not None:
                        A = global_components['A']
                        if hasattr(A, 'toarray'):
                            A_dense = A.toarray()
                        else:
                            A_dense = A
                        comp_group.create_dataset('A', data=A_dense)
                    
                    # Save temporal components (C)
                    if global_components.get('C') is not None:
                        comp_group.create_dataset('C', data=global_components['C'])
                    
                    # Save background components
                    if global_components.get('b') is not None:
                        b = global_components['b']
                        if hasattr(b, 'toarray'):
                            b_dense = b.toarray()
                        else:
                            b_dense = b
                        comp_group.create_dataset('b', data=b_dense)
                    
                    if global_components.get('f') is not None:
                        comp_group.create_dataset('f', data=global_components['f'])
                    
                    # Save deconvolved spikes if available
                    if global_components.get('S') is not None:
                        comp_group.create_dataset('S', data=global_components['S'])
                    
                    # Save metadata
                    meta_group = f.create_group('metadata')
                    meta_group.attrs['n_components'] = global_components['C'].shape[0] if global_components.get('C') is not None else 0
                    meta_group.attrs['n_frames'] = data.shape[0]
                    meta_group.attrs['height'] = data.shape[1]
                    meta_group.attrs['width'] = data.shape[2]
                    meta_group.attrs['source_file'] = Path(tif_path).name
            except Exception as e:
                logger.warning(f"Could not save CNMF-E components: {str(e)}")
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
    if apply_cnmf and correction_method not in ["cnmf", "cnmf_e"]:
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
        meta_group.attrs['n_frames'] = int(n_frames)
        meta_group.attrs['correction_method'] = str(correction_method)
        
        if correction_method == "polynomial_detrend":
            meta_group.attrs['polynomial_order'] = int(polynomial_order)
        if correction_method == "detrend_movie":
            meta_group.attrs['detrend_degree'] = int(config.get("polynomial_order", 1))
        if correction_method == "binomial_exponential":
            meta_group.attrs['binomial_order'] = int(config.get("binomial_order", 3))
        if correction_method == "lowpass_filter":
            meta_group.attrs['cutoff_freq'] = float(config.get("cutoff_freq", 0.001))
        
        meta_group.attrs['apply_cnmf'] = bool(apply_cnmf)
        
        # Handle num_components safely
        if apply_cnmf or correction_method in ["cnmf", "cnmf_e"]:
            if num_components is None:
                meta_group.attrs['num_components'] = -1  # Use -1 to indicate None
            else:
                try:
                    meta_group.attrs['num_components'] = int(num_components)
                except (TypeError, ValueError):
                    # If it can't be converted to int, store as string
                    meta_group.attrs['num_components_str'] = str(num_components)
        
        meta_group.attrs['smoothing_sigma'] = float(smoothing_sigma)
        meta_group.attrs['gpu_accelerated'] = bool(use_gpu and correction_method != "cnmf")
        meta_group.attrs['source_file'] = str(Path(tif_path).name)
    
    # Create a verification plot
    try:
        # Create verification plot
        plt.figure(figsize=(12, 6))
        
        # Plot mean intensity over time
        plt.plot(np.mean(data, axis=(1, 2)), 'r-', alpha=0.7, label='Original')
        plt.plot(np.mean(final_data, axis=(1, 2)), 'g-', alpha=0.7, label=f'{correction_method.replace("_", " ").title()}' + 
                (' + CNMF' if apply_cnmf and correction_method not in ["cnmf", "cnmf_e"] else ''))
        
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