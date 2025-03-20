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

def spatial_downsample_and_detrend(
    data, 
    downsample_factor=2, 
    poly_order=3,
    smoothing_sigma=2.0, 
    use_gpu=False, 
    logger=None
):
    """
    Correct photobleaching with spatial downsampling and polynomial detrending.
    
    Implementation following the approach:
    "To increase the SNR and improve the processing speed, we spatially downsampled 
    each frame in the x and y lateral dimensions by conducting 2 × 2 or 4 × 4 
    bilinear interpolation. To account for photobleaching during imaging, we detrended 
    the Ca2+ movies by calculating the mean for each frame, temporally ordered them 
    from the first to last frame and then fit a first- or third-order polynomial curve 
    to the fluorescent values. We then subtracted all pixels values in a given frame 
    from the fitted values at each time point and added the mean of all pixels in the movie, 
    which detrended the movie while keeping the intensity values in a similar range as 
    the raw movie and prevented the introduction of negative values."
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    downsample_factor : int
        Factor for spatial downsampling (2 for 2×2, 4 for 4×4)
    poly_order : int
        Order of polynomial for detrending (1 for linear, 3 for cubic)
    smoothing_sigma : float
        Sigma for optional Gaussian smoothing after correction
    use_gpu : bool
        Whether to use GPU acceleration if available
    logger : logging.Logger, optional
        Logger object for progress tracking
        
    Returns
    -------
    numpy.ndarray
        Corrected and downsampled fluorescence data
    """
    import numpy as np
    from skimage.transform import resize
    import time
    from scipy.ndimage import gaussian_filter
    
    if logger:
        logger.info(f"Starting spatial downsampling and detrending")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Downsample factor: {downsample_factor}x")
        logger.info(f"Polynomial order: {poly_order}")
    
    start_time = time.time()
    
    # Original dimensions
    n_frames, height, width = data.shape
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Step 1: Spatial downsampling with bilinear interpolation
    if downsample_factor > 1:
        if logger:
            logger.info(f"Performing {downsample_factor}×{downsample_factor} spatial downsampling")
        
        # Calculate new dimensions
        new_height = height // downsample_factor
        new_width = width // downsample_factor
        
        # Preallocate downsampled array
        downsampled = np.zeros((n_frames, new_height, new_width), dtype=np.float32)
        
        # Downsample each frame with bilinear interpolation
        for i in range(n_frames):
            # resize with bilinear interpolation (order=1 specifies bilinear)
            downsampled[i] = resize(data[i], (new_height, new_width), 
                                   order=1, anti_aliasing=True, preserve_range=True)
            
            # Log progress periodically
            if logger and i % 100 == 0:
                logger.info(f"Downsampled {i}/{n_frames} frames")
        
        if logger:
            logger.info(f"Downsampling complete. New shape: {downsampled.shape}")
        
        data = downsampled
    
    # Step 2: Polynomial detrending for photobleaching correction
    if logger:
        logger.info(f"Applying polynomial detrending (order={poly_order})")
    
    # Calculate mean fluorescence of each frame
    mean_per_frame = np.mean(data, axis=(1, 2))
    
    # Timepoints (frame indices)
    t = np.arange(n_frames)
    
    # Fit polynomial to mean fluorescence values
    poly_coeffs = np.polyfit(t, mean_per_frame, poly_order)
    fitted_trend = np.polyval(poly_coeffs, t)
    
    # Calculate global mean across all pixels and frames
    global_mean = float(np.mean(data))
    
    if logger:
        logger.info(f"Fitted {poly_order}-order polynomial for detrending")
        logger.info(f"Global mean intensity: {global_mean:.4f}")
    
    # Apply correction as described:
    # "We then subtracted all pixels values in a given frame from the fitted values 
    # at each time point and added the mean of all pixels in the movie"
    corrected_data = np.zeros_like(data)
    
    for i in range(n_frames):
        # For each frame: fitted_value - frame_data + global_mean
        corrected_data[i] = fitted_trend[i] - data[i] + global_mean
    
    # Apply optional Gaussian smoothing
    if smoothing_sigma > 0:
        if logger:
            logger.info(f"Applying Gaussian smoothing (sigma={smoothing_sigma})")
        
        for i in range(n_frames):
            corrected_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
    
    if logger:
        elapsed_time = time.time() - start_time
        logger.info(f"Spatial downsampling and detrending completed in {elapsed_time:.2f} seconds")
    
    return corrected_data

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
    
    downsample_factor = config.get("downsample_factor", 2)  # Default to 2×2 downsampling

    logger.info(f"Using correction method: {correction_method}")
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Apply photobleaching correction based on method
    if correction_method == "downsample_and_detrend":
        # Use the new combined method
        corrected_data = spatial_downsample_and_detrend(
            data,
            downsample_factor=downsample_factor,
            poly_order=polynomial_order,
            smoothing_sigma=smoothing_sigma,
            use_gpu=use_gpu,
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

def lowpass_filter_correction(data, cutoff_freq=0.001, smoothing_sigma=2.0, temporal_filter_order=2,
                            spatial_hp_sigma=0.0, median_filter_size=3, return_background=False, 
                            memory_efficient=True, max_chunk_size=100, logger=None):
    """
    Enhanced background subtraction using low-pass filtering method with memory efficiency.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    cutoff_freq : float
        Cutoff frequency for low-pass filter (lower values = more aggressive correction)
    smoothing_sigma : float
        Sigma for final Gaussian smoothing
    temporal_filter_order : int
        Order of the Butterworth filter for temporal filtering
    spatial_hp_sigma : float
        Sigma for spatial high-pass filter (0 to disable)
    median_filter_size : int
        Size of median filter for preprocessing (0 to disable)
    return_background : bool
        Whether to return the estimated background in addition to corrected data
    memory_efficient : bool
        Whether to use memory-efficient processing for large data
    max_chunk_size : int
        Maximum number of frames to process at once in memory-efficient mode
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray or tuple
        Corrected fluorescence data or (corrected_data, background)
    """
    if logger:
        logger.info(f"Applying enhanced low-pass filter correction (cutoff_freq={cutoff_freq})")
        data_size_gb = data.nbytes / (1024**3)
        logger.info(f"Input data size: {data_size_gb:.2f} GB")
        if memory_efficient and data_size_gb > 1.0:
            logger.info(f"Using memory-efficient processing with chunks of {max_chunk_size} frames")
    
    # Convert to float32 to avoid dtype issues
    data = data.astype(np.float32)
    n_frames, height, width = data.shape
    
    # Apply median filter to correct bad pixels if requested
    if median_filter_size > 0:
        if memory_efficient:
            # Process in chunks to reduce memory usage
            median_filtered = np.zeros_like(data, dtype=np.float32)
            for i in range(0, n_frames, max_chunk_size):
                end_idx = min(i + max_chunk_size, n_frames)
                for j in range(i, end_idx):
                    median_filtered[j] = median_filter(data[j], size=median_filter_size)
                if logger and i % 100 == 0:
                    logger.info(f"Applied median filter to frames {i} - {end_idx-1}")
        else:
            median_filtered = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[0]):
                median_filtered[i] = median_filter(data[i], size=median_filter_size)
        
        if logger:
            logger.info(f"Applied median filter (size={median_filter_size}) to correct bad pixels")
    else:
        median_filtered = data
    
    # Apply spatial high-pass filter to enhance cellular features if requested
    if spatial_hp_sigma > 0:
        if memory_efficient:
            # Process in chunks
            spatially_filtered = np.zeros_like(median_filtered, dtype=np.float32)
            for i in range(0, n_frames, max_chunk_size):
                end_idx = min(i + max_chunk_size, n_frames)
                for j in range(i, end_idx):
                    # Create low-pass version by gaussian filtering
                    low_pass = gaussian_filter(median_filtered[j], sigma=spatial_hp_sigma)
                    # Subtract to get high-pass filtered (enhance local features)
                    spatially_filtered[j] = median_filtered[j] - low_pass
                if logger and i % 100 == 0:
                    logger.info(f"Applied spatial high-pass filter to frames {i} - {end_idx-1}")
        else:
            spatially_filtered = np.zeros_like(median_filtered, dtype=np.float32)
            for i in range(n_frames):
                # Create low-pass version by gaussian filtering
                low_pass = gaussian_filter(median_filtered[i], sigma=spatial_hp_sigma)
                # Subtract to get high-pass filtered (enhance local features)
                spatially_filtered[i] = median_filtered[i] - low_pass
            
        if logger:
            logger.info(f"Applied spatial high-pass filter (sigma={spatial_hp_sigma}) to enhance cell features")
            
        # For temporal filtering, we'll use the median filtered data without spatial high-pass
        data_for_temporal = median_filtered
    else:
        spatially_filtered = median_filtered
        data_for_temporal = median_filtered
    
    # Apply low-pass filter to capture slow changes (photobleaching, illumination drift)
    # Design Butterworth low-pass filter
    b, a = signal.butter(temporal_filter_order, cutoff_freq, 'low')
    
    # In memory-efficient mode, process each pixel separately to avoid large reshapes
    if memory_efficient:
        if logger:
            logger.info("Starting memory-efficient temporal filtering (this may take a while)")
        
        # Create the background model (low-pass filtered data)
        lowpass_data = np.zeros_like(data_for_temporal, dtype=np.float32)
        
        # Process each pixel individually to avoid large memory allocations
        for y in range(height):
            for x in range(width):
                # Extract time series for this pixel
                pixel_timeseries = data_for_temporal[:, y, x]
                # Apply filter
                lowpass_data[:, y, x] = signal.filtfilt(b, a, pixel_timeseries)
            
            # Log progress
            if logger and y % 100 == 0:
                logger.info(f"Processed temporal filtering for row {y}/{height}")
    else:
        # Reshape data for filtering (less memory efficient but faster)
        reshaped_data = data_for_temporal.reshape(n_frames, -1)
        
        # Apply filter along time dimension for each pixel
        lowpass_data = np.zeros_like(reshaped_data, dtype=np.float32)
        for i in range(reshaped_data.shape[1]):
            lowpass_data[:, i] = signal.filtfilt(b, a, reshaped_data[:, i])
        
        # Reshape back to original dimensions
        lowpass_data = lowpass_data.reshape(n_frames, height, width)
    
    if logger:
        logger.info(f"Applied temporal low-pass filter (order={temporal_filter_order}, cutoff={cutoff_freq})")
    
    # Calculate global mean (to add back later)
    global_mean = float(np.mean(median_filtered))
    if logger:
        logger.info(f"Global mean intensity: {global_mean:.4f}")
    
    # Subtract low-pass filtered data from filtered data to remove slow trends
    # Use in-place operations to save memory
    if memory_efficient:
        # Process in chunks
        corrected_data = np.zeros_like(spatially_filtered, dtype=np.float32)
        for i in range(0, n_frames, max_chunk_size):
            end_idx = min(i + max_chunk_size, n_frames)
            # Subtract background and add mean in one step to avoid extra array allocation
            corrected_data[i:end_idx] = spatially_filtered[i:end_idx] - lowpass_data[i:end_idx] + global_mean
            
            if logger and i % 100 == 0:
                logger.info(f"Applied background correction to frames {i} - {end_idx-1}")
    else:
        # Perform operations in one step
        corrected_data = spatially_filtered - lowpass_data + global_mean
    
    if logger:
        logger.info(f"Subtracted low-pass trend and added global mean")
    
    # Apply final Gaussian smoothing if requested
    if smoothing_sigma > 0:
        if memory_efficient:
            # In-place smoothing
            for i in range(0, n_frames, max_chunk_size):
                end_idx = min(i + max_chunk_size, n_frames)
                for j in range(i, end_idx):
                    # Create temporary smoothed frame
                    smoothed = gaussian_filter(corrected_data[j], sigma=smoothing_sigma)
                    # Copy back to original array
                    corrected_data[j] = smoothed
                
                if logger and i % 100 == 0:
                    logger.info(f"Applied final smoothing to frames {i} - {end_idx-1}")
        else:
            smoothed_data = np.zeros_like(corrected_data, dtype=np.float32)
            for i in range(n_frames):
                smoothed_data[i] = gaussian_filter(corrected_data[i], sigma=smoothing_sigma)
            corrected_data = smoothed_data
        
        if logger:
            logger.info(f"Applied final Gaussian smoothing (sigma={smoothing_sigma})")
    
    # Return results
    if return_background:
        return corrected_data, lowpass_data
    else:
        return corrected_data

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
        
        # For CaImAn, use the correct parameter naming
        params_dict = {
            'dims': data.shape[1:],          # Image dimensions
            'method_init': 'greedy_roi',     # Initialization method
            'K': num_components,             # Number of components
            'gSig': [4, 4],                  # Expected size of neurons (pixels)
            'p': 1,                          # Order of AR model for calcium dynamics
            'gnb': 2,                        # Number of background components
            'merge_thresh': 0.8,             # Merging threshold
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
        
        # Get components using helper function
        components = get_cnmf_components(cnm, logger=logger)
        
        if 'A' not in components or 'C' not in components or 'b' not in components or 'f' not in components:
            raise ValueError("Could not extract required CNMF components")
        
        # Get background components (contains photobleaching)
        background = components['b'].dot(components['f'])
        
        # Get neural components (denoised signals)
        neural = components['A'].dot(components['C'])
        
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

def cnmf_e_correction(
    data, 
    masks=None,
    num_components=None, 
    min_corr=0.8,
    min_pnr=10,
    gSig=(4, 4),
    gSiz=(15, 15),
    ring_size_factor=1.5,
    merge_thresh=0.8,
    use_multiprocessing=False,
    n_processes=1,
    logger=None
):
    """
    Enhanced CNMF-E for microendoscopic data with support for pre-defined ROIs.
    Version optimized for CaImAn 1.8+ compatibility.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input fluorescence data with shape (frames, height, width)
    masks : numpy.ndarray, optional
        Pre-defined binary masks with shape (n_masks, height, width) for initialization
    num_components : int, optional
        Number of neural components to extract (if None, estimated automatically)
    min_corr : float, optional
        Minimum correlation for component detection
    min_pnr : float, optional
        Minimum peak-to-noise ratio for component detection
    gSig : tuple, optional
        Expected half-size of neurons
    gSiz : tuple, optional
        Expected full size of neurons
    ring_size_factor : float, optional
        Factor to enlarge the ring for background estimation
    merge_thresh : float, optional
        Merging threshold for components
    use_multiprocessing : bool, optional
        Whether to use multiple processes
    n_processes : int, optional
        Number of processes to use if multiprocessing
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Corrected fluorescence data with shape (frames, height, width)
    """
    try:
        import caiman as cm
        from caiman.source_extraction.cnmf import cnmf as cnmf
        from caiman.source_extraction.cnmf import params as params
        from scipy import sparse
        
        start_time = time.time()
        
        if logger:
            logger.info("Starting enhanced CNMF-E analysis for microendoscopic data")
            logger.info(f"Using CaImAn version: {cm.__version__}")
        
        # Convert data to float32
        Y = data.astype(np.float32)
        
        # Get dimensions
        T, d1, d2 = data.shape
        dims = (d1, d2)
        
        # Check if we have pre-defined masks
        has_masks = masks is not None and masks.shape[0] > 0
        
        if has_masks:
            if logger:
                logger.info(f"Using {masks.shape[0]} pre-defined ROIs for CNMF-E initialization")
            # Verify mask dimensions
            if masks.shape[1:] != dims:
                if logger:
                    logger.warning(f"Mask dimensions ({masks.shape[1:]}) don't match data dimensions ({dims})")
                    logger.warning("Attempting to resize masks...")
                try:
                    from skimage.transform import resize
                    resized_masks = np.zeros((masks.shape[0], d1, d2), dtype=masks.dtype)
                    for i in range(masks.shape[0]):
                        resized_masks[i] = resize(masks[i], dims, order=0, preserve_range=True)
                    masks = resized_masks
                    if logger:
                        logger.info(f"Successfully resized masks to {masks.shape}")
                except Exception as e:
                    if logger:
                        logger.error(f"Failed to resize masks: {str(e)}")
                    has_masks = False
        
        # Parameters for CNMF-E
        if logger:
            logger.info(f"Setting up CNMF-E parameters (gSig={gSig}, min_corr={min_corr}, min_pnr={min_pnr})")
        
        # Create parameters dictionary
        params_dict = {}
        
        # Dataset parameters
        params_dict['fr'] = 30.0                    # frame rate
        params_dict['decay_time'] = 0.4             # length of a typical transient (seconds)
        
        # CNMF parameters
        params_dict['p'] = 1                        # order of AR model
        params_dict['nb'] = 0                       # number of background components (0 for CNMF-E)
        params_dict['merge_thresh'] = merge_thresh  # merging threshold
        
        # Patch parameters (None = process whole FOV at once if possible)
        if data.shape[1] * data.shape[2] > 512 * 512:
            # For large FOVs, use patches
            params_dict['rf'] = 40                  # half-size of patches
            params_dict['stride'] = 20              # amount of overlap between patches
        else:
            # For smaller FOVs, process whole image
            params_dict['rf'] = None
            params_dict['stride'] = None
        
        # Component detection parameters
        params_dict['K'] = num_components if num_components is not None else None  # number of components per patch
        params_dict['gSig'] = gSig                  # expected half size of neurons
        params_dict['gSiz'] = gSiz                  # size of bounding box
        params_dict['min_corr'] = min_corr          # min peak-to-noise ratio
        params_dict['min_pnr'] = min_pnr            # min correlation
        
        # CNMF-E specific parameters
        params_dict['method_init'] = 'corr_pnr'     # initialization method
        params_dict['ring_size_factor'] = ring_size_factor  # ring size factor
        params_dict['center_psf'] = True            # center the PSF
        
        # Spatial parameters
        params_dict['update_background_components'] = True  # update background
        params_dict['ssub'] = 1                     # spatial downsampling
        params_dict['tsub'] = 2                     # temporal downsampling
        
        # Initialize parameters
        cnm_params = params.CNMFParams(params_dict=params_dict)
        
        # Set up dview for parallel processing if requested
        dview = None
        if use_multiprocessing and n_processes > 1:
            try:
                from multiprocessing import Pool
                dview = Pool(n_processes)
                if logger:
                    logger.info(f"Set up multiprocessing pool with {n_processes} workers")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not set up multiprocessing: {str(e)}")
        
        # Create CNMF object
        cnm = cnmf.CNMF(n_processes=n_processes, params=cnm_params, dview=dview)
        
        # Initialize with pre-defined masks if available
        if has_masks:
            if logger:
                logger.info("Preparing pre-defined masks for CNMF-E initialization")
            
            # Reshape masks to pixels × components
            n_masks = masks.shape[0]
            masks_flat = masks.reshape(n_masks, -1).T  # Transpose to match CaImAn's format
            
            # Normalize masks if needed
            for i in range(n_masks):
                col_max = np.max(masks_flat[:, i])
                if col_max > 0:
                    masks_flat[:, i] = masks_flat[:, i] / col_max
            
            # Convert to sparse matrix
            A_init = sparse.csc_matrix(masks_flat)
            
            if logger:
                logger.info(f"Created initial A matrix with shape {A_init.shape}")
            
            # Pre-allocate estimates for initialization
            # Handle different CaImAn versions
            try:
                # New versions with estimates
                if hasattr(cnm, 'estimates'):
                    cnm.estimates = cnmf.Estimates(A=A_init, dims=dims)
                else:
                    # Old versions with direct attribute assignment
                    cnm.A = A_init
                    cnm.dims = dims
                
                if logger:
                    logger.info("Successfully initialized CNMF-E with pre-defined masks")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not initialize with masks: {str(e)}")
                has_masks = False
            
            # Set flag to use initialization
            use_init = has_masks
        else:
            if logger:
                logger.info("No valid masks provided, will initialize from scratch")
            use_init = False
        
        # Fit the model
        if logger:
            logger.info("Running CNMF-E fit")
            
        if use_init:
            # With pre-initialized components
            try:
                cnm = cnm.fit(Y, init_batch=0)
            except:
                # Fall back to standard fit if init_batch fails
                cnm = cnm.fit(Y)
        else:
            # Standard initialization
            cnm = cnm.fit(Y)
        
        # Get components using helper function
        components = get_cnmf_components(cnm, logger=logger)
        
        # Check if we have components
        if 'A' not in components or 'C' not in components:
            if logger:
                logger.warning("CNMF-E did not find any usable components")
            return Y  # Return original data if no components
        
        # Extract number of components
        n_components = components['A'].shape[1] if hasattr(components['A'], 'shape') else 0
        
        if logger:
            logger.info(f"CNMF-E found {n_components} components")
        
        # Get indices of good components if available
        idx_components = components.get('idx_components', None)
        
        # If no good components specified, try to evaluate them
        if idx_components is None or len(idx_components) == 0:
            try:
                # Set quality threshold parameters
                cnm.params.set('quality', {'min_SNR': 2.0,      # minimum SNR threshold
                                         'rval_thr': 0.85,     # correlation threshold
                                         'use_cnn': False})    # don't use CNN (not calibrated for 1p)
                
                # Evaluate components
                cnm.estimates.evaluate_components(Y, cnm.params)
                
                # Update components with evaluation results
                components = get_cnmf_components(cnm, logger=logger)
                idx_components = components.get('idx_components', None)
                
                if logger and idx_components is not None:
                    logger.info(f"Component quality evaluation: {len(idx_components)} accepted")
            except Exception as e:
                if logger:
                    logger.warning(f"Component evaluation failed: {str(e)}")
                # If evaluation fails, use all components
                idx_components = None
        
        # If still no good components, use all
        if idx_components is None or len(idx_components) == 0:
            idx_components = np.arange(n_components)
            
            if logger:
                logger.info(f"Using all {n_components} components (no quality filtering)")
        
        # Extract dimensions
        n_pixels = d1 * d2
        
        # Reconstruct data
        try:
            # Get neural components (only good ones if available)
            if len(idx_components) > 0:
                A_good = components['A'][:, idx_components]
                C_good = components['C'][idx_components]
                AC = A_good.dot(C_good)
            else:
                # If no good components, use all
                AC = components['A'].dot(components['C'])
            
            # Get background components
            try:
                if 'b' in components and 'f' in components:
                    bf = components['b'].dot(components['f'])
                else:
                    if logger:
                        logger.warning("Missing background components, using zeros")
                    bf = np.zeros((n_pixels, T))
            except Exception as e:
                if logger:
                    logger.warning(f"Error getting background components: {str(e)}")
                bf = np.zeros((n_pixels, T))
            
            # Combine neural and background
            Y_rec = AC + bf
            
            # Reshape to original dimensions
            if Y_rec.shape[0] == n_pixels:
                # Shape is (pixels, frames)
                corrected_data = Y_rec.reshape((d1, d2, T), order='F')
                corrected_data = np.moveaxis(corrected_data, -1, 0)
            else:
                if logger:
                    logger.warning(f"Unexpected component shape: {Y_rec.shape}, expected first dim to be {n_pixels}")
                return Y
        except Exception as e:
            if logger:
                logger.error(f"Error reconstructing data: {str(e)}")
            return Y
        
        # Store the components for future use
        global cnmf_e_components
        cnmf_e_components = components
        
        # Clean up
        if dview is not None:
            dview.terminate()
        
        if logger:
            logger.info(f"CNMF-E processing completed in {time.time() - start_time:.2f} seconds")
            
        return corrected_data
        
    except ImportError as e:
        if logger:
            logger.error(f"CaImAn import failed: {str(e)}")
            logger.warning("CNMF-E requires CaImAn. Please install it with: pip install caiman")
        raise
        
    except Exception as e:
        if logger:
            logger.error(f"CNMF-E processing failed: {str(e)}")
        raise