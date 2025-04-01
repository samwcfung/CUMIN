#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Processing Module
------------------
Handles ROI extraction and background subtraction using polygon-based masks.
"""

import os
import time
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_dilation, median_filter
from scipy import signal
from skimage.draw import polygon
from skimage.transform import resize
import pandas as pd

def extract_roi_fluorescence(
    roi_path, 
    image_data, 
    image_shape, 
    output_dir, 
    config, 
    logger
):
    """
    Extract ROI fluorescence traces from an image stack.
    """
    start_time = time.time()
    logger.info(f"Extracting ROIs from {Path(roi_path).name}")
    
    # Create directory for intermediate traces if enabled
    trace_dir = os.path.join(output_dir, config.get("intermediate_trace_dir", "intermediate_traces"))
    save_intermediate = config.get("save_intermediate_traces", False)
    if save_intermediate:
        os.makedirs(trace_dir, exist_ok=True)
        logger.info(f"Will save intermediate traces to {trace_dir}")
    
    # Check if we should perform the extraction step
    if not config.get("steps", {}).get("extract_rois", True):
        logger.info("ROI extraction step is disabled in config. Skipping.")
        return [], None
    
    # Extract ROIs from zip file
    roi_masks, roi_centers = extract_rois_from_zip(roi_path, image_shape, logger)
    
    # If no valid ROIs were found, log error and return empty results
    if not roi_masks:
        error_msg = f"No valid ROIs were found in {Path(roi_path).name}. Skipping ROI extraction."
        logger.error(error_msg)
        return [], None
    
    # Save ROI masks as a reference PNG if enabled
    if config.get("steps", {}).get("save_masks", True):
        save_roi_visualization(roi_masks, image_shape, output_dir, logger)
        
        # Save masks in NPY format
        mask_array = np.stack(roi_masks)
        np_path = os.path.join(output_dir, "roi_masks.npy")
        np.save(np_path, mask_array)
        logger.info(f"Saved binary masks to {np_path}")
        
        # Save masks for CNMF processing
        save_masks_for_cnmf(roi_masks, output_dir, logger)
    
    # Extract fluorescence traces
    if image_data is not None:
        logger.info(f"Extracting fluorescence traces for {len(roi_masks)} ROIs")
        roi_fluorescence = extract_traces(image_data, roi_masks, logger)
        
        # Save raw traces if enabled
        if save_intermediate:
            raw_traces_path = os.path.join(trace_dir, "1_extracted_raw_traces.csv")
            save_trace_data(roi_fluorescence, raw_traces_path, config.get("trace_save_format", "csv"))
            logger.info(f"Saved extracted raw traces to {raw_traces_path}")
        
        # Apply ROI-specific detrending if enabled
        if config.get("steps", {}).get("roi_specific_detrend", False) and config.get("roi_detrend", {}).get("enabled", False):
            logger.info("Applying ROI-specific detrending")
            
            # Store original traces for comparison
            original_traces = roi_fluorescence.copy()
            
            # Apply detrending
            polynomial_degree = config["roi_detrend"].get("polynomial_degree", 1)
            roi_fluorescence = roi_specific_detrend(roi_fluorescence, polynomial_degree, logger)
            
            # Save detrended traces if enabled
            if save_intermediate:
                detrended_path = os.path.join(trace_dir, "2_roi_detrended_traces.csv")
                save_trace_data(roi_fluorescence, detrended_path, config.get("trace_save_format", "csv"))
                logger.info(f"Saved ROI-detrended traces to {detrended_path}")
    else:
        logger.warning("No image data provided, skipping fluorescence extraction")
        roi_fluorescence = None
    
    logger.info(f"ROI extraction completed in {time.time() - start_time:.2f} seconds")
    
    return roi_masks, roi_fluorescence

def save_trace_data(trace_data, path, format="csv"):
    """
    Save trace data to file in specified format.
    
    Parameters
    ----------
    trace_data : numpy.ndarray
        Trace data to save
    path : str
        Path to save the data
    format : str
        Format to save ('csv' or 'h5')
    """
    if format.lower() == "csv":
        pd.DataFrame(trace_data).to_csv(path)
    elif format.lower() == "h5":
        with h5py.File(path, 'w') as f:
            f.create_dataset('traces', data=trace_data)
    else:
        # Default to CSV
        pd.DataFrame(trace_data).to_csv(path)

def extract_rois_from_zip(roi_path, image_shape, logger):
    """
    Extract ROI masks from ImageJ/FIJI .zip ROI file using polygon-based approach.
    """
    roi_masks = []
    roi_centers = []
    
    try:
        # Try to import the ROI reader
        try:
            from roifile import roiread
            logger.info("Using roifile library for ROI extraction")
        except ImportError:
            logger.warning("roifile library not found, trying read_roi package")
            try:
                import read_roi
                logger.info("Using read_roi package for ROI extraction")
            except ImportError:
                logger.error("ROI reading libraries not found. Please install with: pip install roifile read-roi")
                return [], []
        
        # First try using roifile if available
        try:
            from roifile import roiread
            logger.info("Extracting ROIs with roifile library")
            
            # Read all ROIs from the file
            rois = roiread(roi_path)
            logger.info(f"Found {len(rois)} ROIs using roifile library")
            
            for i, roi in enumerate(rois):
                try:
                    # Create empty mask
                    mask = np.zeros(image_shape, dtype=bool)
                    
                    # Get coordinates
                    coords = roi.coordinates()
                    
                    # Check if we have valid coordinates
                    if coords.size == 0 or coords.shape[0] < 3:  # Need at least 3 points for a polygon
                        logger.warning(f"ROI {i+1} has insufficient coordinates, skipping")
                        continue
                    
                    # Ensure coordinates are within image bounds
                    coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[1]-1)
                    coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[0]-1)
                    
                    # Use skimage's polygon function to create mask
                    try:
                        rr, cc = polygon(coords[:, 1], coords[:, 0], image_shape)
                        mask[rr, cc] = True
                    except Exception as poly_e:
                        logger.warning(f"Error creating polygon for ROI {i+1}: {str(poly_e)}")
                        continue
                    
                    # Calculate ROI center
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        center_y = int(np.mean(y_indices))
                        center_x = int(np.mean(x_indices))
                        
                        # Add mask and center to lists
                        roi_masks.append(mask)
                        roi_centers.append((center_y, center_x))
                    else:
                        logger.warning(f"ROI {i+1} has an empty mask, skipping")
                        
                except Exception as e:
                    logger.warning(f"Error processing ROI {i+1}: {str(e)}")
            
            # If we got any ROIs, return them
            if roi_masks:
                return roi_masks, roi_centers
            
        except (ImportError, Exception) as e:
            logger.warning(f"Error using roifile: {str(e)}")
        
        # Try with read_roi if roifile failed
        try:
            import read_roi
            logger.info("Extracting ROIs with read_roi package")
            
            # Read all ROIs from the zip file
            roi_dict = read_roi.read_roi_zip(roi_path)
            
            if roi_dict:
                logger.info(f"Found {len(roi_dict)} ROIs using read_roi package")
                
                for name, roi in roi_dict.items():
                    try:
                        # Create empty mask
                        mask = np.zeros(image_shape, dtype=bool)
                        
                        # Check if we have coordinates for a polygon
                        if 'x' in roi and 'y' in roi:
                            x = np.array(roi['x'])
                            y = np.array(roi['y'])
                            
                            # Check if we have valid coordinates
                            if len(x) < 3 or len(y) < 3:  # Need at least 3 points for a polygon
                                logger.warning(f"ROI {name} has insufficient coordinates, skipping")
                                continue
                            
                            # Ensure coordinates are within image bounds
                            x = np.clip(x, 0, image_shape[1]-1)
                            y = np.clip(y, 0, image_shape[0]-1)
                            
                            # Use skimage's polygon function to create mask
                            try:
                                rr, cc = polygon(y, x, image_shape)
                                mask[rr, cc] = True
                            except Exception as poly_e:
                                logger.warning(f"Error creating polygon for ROI {name}: {str(poly_e)}")
                                continue
                                
                        elif 'left' in roi and 'top' in roi and 'width' in roi and 'height' in roi:
                            # For rectangle/oval ROIs
                            left = max(0, min(roi['left'], image_shape[1]-1))
                            top = max(0, min(roi['top'], image_shape[0]-1))
                            width = min(roi['width'], image_shape[1] - left)
                            height = min(roi['height'], image_shape[0] - top)
                            
                            if width <= 0 or height <= 0:
                                logger.warning(f"ROI {name} has invalid dimensions, skipping")
                                continue
                                
                            # Check if it's an oval
                            if roi.get('type', 'rectangle') == 'oval':
                                # For oval ROIs
                                center = (int(left + width / 2), int(top + height / 2))
                                axes = (int(width / 2), int(height / 2))
                                mask_temp = np.zeros(image_shape, dtype=np.uint8)
                                cv2.ellipse(mask_temp, center, axes, 0, 0, 360, 1, -1)
                                mask = mask_temp.astype(bool)
                            else:
                                # For rectangle ROIs
                                mask[top:top+height, left:left+width] = True
                        else:
                            logger.warning(f"Unsupported ROI format: {name}")
                            continue
                            
                        # Calculate ROI center
                        y_indices, x_indices = np.where(mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            center_y = int(np.mean(y_indices))
                            center_x = int(np.mean(x_indices))
                            
                            # Add mask and center to lists
                            roi_masks.append(mask)
                            roi_centers.append((center_y, center_x))
                        else:
                            logger.warning(f"ROI {name} has an empty mask, skipping")
                            
                    except Exception as e:
                        logger.warning(f"Error processing ROI {name}: {str(e)}")
                
                # If we got any ROIs, return them
                if roi_masks:
                    return roi_masks, roi_centers
            
        except (ImportError, Exception) as e:
            logger.warning(f"Error using read_roi: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error extracting ROIs from {roi_path}: {str(e)}", exc_info=True)
    
    if roi_masks:
        logger.info(f"Successfully extracted {len(roi_masks)} ROI masks")
    else:
        logger.warning("No ROI masks were successfully extracted")
    
    return roi_masks, roi_centers

def save_roi_visualization(roi_masks, image_shape, output_dir, logger):
    """
    Save ROI masks as a PNG visualization.
    """
    # Check if we have any masks
    if not roi_masks:
        logger.warning("No ROI masks to visualize")
        return
    
    # Create a composite mask with all ROIs
    composite_mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Assign different colors to each ROI
    for i, mask in enumerate(roi_masks):
        # Add ROI to composite with unique intensity
        composite_mask[mask] = i + 1
    
    # Create a color-coded visualization
    vis_image = np.zeros((*image_shape, 3), dtype=np.uint8)
    
    # Generate random colors for each ROI
    np.random.seed(0)  # For reproducibility
    colors = np.random.randint(50, 255, size=(len(roi_masks), 3))
    
    # Apply colors to ROIs
    for i in range(len(roi_masks)):
        roi_indices = composite_mask == (i + 1)
        vis_image[roi_indices] = colors[i]
    
    # Add ROI numbers
    for i, mask in enumerate(roi_masks):
        # Find centroid of ROI
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            
            # Add text to indicate ROI number
            cv2.putText(vis_image, str(i+1), (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Save visualization
    vis_path = os.path.join(output_dir, "roi_masks.png")
    cv2.imwrite(vis_path, vis_image)
    
    logger.info(f"Saved ROI visualization to {vis_path}")
    
    # Also save individual mask images
    masks_dir = os.path.join(output_dir, "individual_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for i, mask in enumerate(roi_masks):
        # Convert binary mask to uint8
        mask_img = mask.astype(np.uint8) * 255
        
        # Save mask
        mask_path = os.path.join(masks_dir, f"mask_{i+1:03d}.png")
        cv2.imwrite(mask_path, mask_img)
    
    logger.info(f"Saved {len(roi_masks)} individual mask images to {masks_dir}")

def save_masks_for_cnmf(roi_masks, output_dir, logger):
    """
    Save ROI masks in formats suitable for CNMF/CNMF-E processing.
    
    Parameters
    ----------
    roi_masks : list
        List of ROI masks
    output_dir : str
        Directory to save the masks
    logger : logging.Logger
        Logger object
    
    Returns
    -------
    str
        Path to the saved masks file
    """
    # Check if we have any masks
    if not roi_masks:
        logger.warning("No ROI masks to save for CNMF")
        return None
    
    # Create directory for masks if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert list of masks to 3D array (n_masks, height, width)
    mask_array = np.stack(roi_masks)
    
    # Save as HDF5 file which is compatible with CNMF
    h5_path = os.path.join(output_dir, "roi_masks_for_cnmf.h5")
    
    with h5py.File(h5_path, 'w') as f:
        # Create dataset for masks
        f.create_dataset('masks', data=mask_array.astype(np.float32))
        f.attrs['n_masks'] = len(roi_masks)
    
    logger.info(f"Saved {len(roi_masks)} masks for CNMF/CNMF-E to {h5_path}")
    
    return h5_path

def extract_traces(image_data, roi_masks, logger=None):
    """
    Extract fluorescence traces from ROIs in an image stack.
    """
    n_rois = len(roi_masks)
    n_frames = image_data.shape[0]
    
    # Get dimensions
    _, img_height, img_width = image_data.shape
    
    # SAFETY CHECK: Ensure masks are numpy arrays
    for i in range(len(roi_masks)):
        if not isinstance(roi_masks[i], np.ndarray):
            roi_masks[i] = np.array(roi_masks[i], dtype=bool)
    
    # Get mask dimensions - SAFELY
    try:
        mask_height, mask_width = roi_masks[0].shape
    except:
        # Fallback in case of error
        if logger:
            logger.warning("Could not determine mask shape, attempting to infer")
        mask_height = 0
        mask_width = 0
        for mask in roi_masks:
            if hasattr(mask, 'shape') and len(mask.shape) == 2:
                mask_height, mask_width = mask.shape
                break
    
    # Check for dimension mismatch and resize masks if needed
    if mask_height != img_height or mask_width != img_width:
        if logger:
            logger.warning(f"ROI mask dimensions ({mask_height}x{mask_width}) don't match image dimensions ({img_height}x{img_width})")
            logger.info(f"Resizing ROI masks to match image dimensions")
        
        # Resize masks to match image dimensions
        resized_masks = []
        for mask in roi_masks:
            # Convert to binary numpy array if needed
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=bool)
            
            # Convert boolean mask to float for resize operation
            mask_float = mask.astype(np.float32)
            
            # Use nearest-neighbor interpolation (order=0) to preserve binary nature
            resized_mask = resize(mask_float, (img_height, img_width), 
                                 order=0, preserve_range=True, anti_aliasing=False)
            
            # Convert back to boolean
            resized_mask = resized_mask > 0.5
            resized_masks.append(resized_mask)
        
        roi_masks = resized_masks
        
        if logger:
            logger.info(f"ROI masks resized to {img_height}x{img_width}")
    
    # Extract fluorescence traces
    traces = np.zeros((n_rois, n_frames), dtype=np.float32)
    
    for i, mask in enumerate(roi_masks):
        for t in range(n_frames):
            # Ensure mask is boolean
            binary_mask = mask.astype(bool)
            # Extract mean value
            traces[i, t] = np.mean(image_data[t][binary_mask])
    
    return traces

def roi_specific_detrend(roi_traces, detrend_degree=1, logger=None):
    """
    Apply polynomial detrending to each ROI trace individually.
    
    Parameters
    ----------
    roi_traces : numpy.ndarray
        ROI traces with shape (n_rois, n_frames)
    detrend_degree : int
        Degree of polynomial for detrending
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Detrended ROI traces
    """
    import numpy as np
    
    if logger:
        logger.info(f"Applying ROI-specific detrending with polynomial degree {detrend_degree}")
    
    n_rois, n_frames = roi_traces.shape
    detrended_traces = np.zeros_like(roi_traces)
    
    # Process each ROI individually
    for i in range(n_rois):
        trace = roi_traces[i]
        t = np.arange(n_frames)
        
        # Fit polynomial
        poly_coeffs = np.polyfit(t, trace, detrend_degree)
        fitted_trend = np.polyval(poly_coeffs, t)
        
        # Calculate mean for this ROI
        roi_mean = np.mean(trace)
        
        # Apply correction: divide by trend and multiply by mean
        # Use np.maximum to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            correction_factor = roi_mean / np.maximum(fitted_trend, 1e-6)
            detrended_traces[i] = trace * correction_factor
        
        # Handle any invalid values (NaN or inf)
        invalid_mask = ~np.isfinite(detrended_traces[i])
        if np.any(invalid_mask):
            detrended_traces[i, invalid_mask] = trace[invalid_mask]
            if logger:
                logger.warning(f"ROI {i+1}: Handled {np.sum(invalid_mask)} invalid values during detrending")
    
    if logger:
        logger.info(f"Successfully detrended {n_rois} ROI traces")
    
    return detrended_traces

def subtract_background(
    image_data,
    roi_data,
    roi_masks,
    bg_config,
    logger,
    output_dir=None
):
    """
    Extract background signal and subtract from ROI fluorescence traces.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Preprocessed image data (frames, height, width)
    roi_data : numpy.ndarray
        ROI fluorescence traces (n_rois, n_frames)
    roi_masks : list of numpy.ndarray
        Binary masks for each ROI
    bg_config : dict
        Background subtraction configuration
    logger : logging.Logger
        Logger object
    output_dir : str, optional
        Output directory for saving intermediate traces
        
    Returns
    -------
    numpy.ndarray
        Background-subtracted ROI traces
    """
    import numpy as np
    import pandas as pd
    import os
    import time
    from scipy.ndimage import binary_dilation
    from scipy.signal import medfilt
    
    start_time = time.time()
    logger.info("Starting background subtraction")
    
    # Get method from config
    method = bg_config.get("method", "darkest_pixels")
    logger.info(f"Using background method: {method}")
    
    # Get number of ROIs and frames
    n_rois, n_frames = roi_data.shape
    
    # Create array to store original ROI data for later comparison
    original_roi_data = roi_data.copy()
    
    # Create array for background traces
    bg_traces = np.zeros((n_rois, n_frames))
    
    # Parameters for extracting background
    if method == "darkest_pixels":
        # Get parameters
        percentile = bg_config.get("percentile", 0.4)
        median_filter_size = bg_config.get("median_filter_size", 5)
        dilation_size = bg_config.get("dilation_size", 2)
        
        # Create background mask
        roi_combined = np.zeros(roi_masks[0].shape, dtype=bool)
        for mask in roi_masks:
            roi_combined = roi_combined | mask
            
        # Dilate ROI mask to avoid pixels close to ROIs
        if dilation_size > 0:
            element = np.ones((dilation_size*2+1, dilation_size*2+1), dtype=bool)
            roi_combined = binary_dilation(roi_combined, element)
            
        # Invert to get background mask
        bg_mask = ~roi_combined
        
        # Extract background trace from darkest pixels percentile
        logger.info(f"Extracting {percentile} percentile background from {np.sum(bg_mask)} background pixels")
        
        # For each frame, find the background value
        bg_values = np.zeros(n_frames)
        for f in range(n_frames):
            frame = image_data[f]
            bg_pixels = frame[bg_mask]
            if len(bg_pixels) > 0:
                bg_values[f] = np.percentile(bg_pixels, percentile * 100)
            else:
                logger.warning(f"No background pixels found in frame {f}")
                # Use the value from previous frame if available
                bg_values[f] = bg_values[f-1] if f > 0 else 0
                
        # Apply median filter to reduce noise in background trace if specified
        if median_filter_size > 0:
            bg_values = medfilt(bg_values, median_filter_size)
            
        # Use the same background trace for all ROIs
        for i in range(n_rois):
            bg_traces[i] = bg_values
            
    elif method == "roi_periphery":
        # Extract background from periphery of each ROI
        periphery_size = bg_config.get("periphery_size", 2)
        logger.info(f"Extracting background from ROI periphery with size {periphery_size}")
        
        for i, mask in enumerate(roi_masks):
            # Dilate mask
            dilated = binary_dilation(mask, np.ones((periphery_size*2+1, periphery_size*2+1), dtype=bool))
            
            # Create periphery mask (dilated - original)
            periphery = dilated & ~mask
            
            # Skip if periphery is empty
            if not np.any(periphery):
                logger.warning(f"ROI {i+1} has empty periphery, using global method")
                # Use the darkest pixels method as fallback
                bg_traces[i] = bg_values if 'bg_values' in locals() else np.zeros(n_frames)
                continue
                
            # Extract background from periphery for each frame
            for f in range(n_frames):
                periphery_pixels = image_data[f][periphery]
                bg_traces[i, f] = np.percentile(periphery_pixels, percentile * 100)
            
            # Apply median filter to reduce noise
            if median_filter_size > 0:
                bg_traces[i] = medfilt(bg_traces[i], median_filter_size)
                
    elif method == "global_background":
        # Handle separately to avoid code duplication
        return subtract_global_background(
            image_data,
            roi_data,
            roi_masks,
            bg_config,
            logger,
            output_dir=output_dir
        )
    
    else:
        logger.warning(f"Unknown background method: {method}, using darkest_pixels")
        # Fallback to darkest_pixels method
        percentile = bg_config.get("percentile", 0.4)
        median_filter_size = bg_config.get("median_filter_size", 5)
        
        # Create global background mask (all pixels)
        bg_mask = np.ones(roi_masks[0].shape, dtype=bool)
        
        # Extract background trace
        bg_values = np.zeros(n_frames)
        for f in range(n_frames):
            frame = image_data[f]
            bg_pixels = frame[bg_mask]
            bg_values[f] = np.percentile(bg_pixels, percentile * 100)
            
        # Apply median filter
        if median_filter_size > 0:
            bg_values = medfilt(bg_values, median_filter_size)
            
        # Use the same background trace for all ROIs
        for i in range(n_rois):
            bg_traces[i] = bg_values
    
    # Save trace before background subtraction
    save_intermediate = bg_config.get("save_intermediate_traces", False)
    if save_intermediate and output_dir:
        traces_dir = os.path.join(output_dir, "intermediate_traces")
        os.makedirs(traces_dir, exist_ok=True)
        
        # Save original traces
        roi_traces_path = os.path.join(traces_dir, "3_roi_traces_before_bg.csv")
        pd.DataFrame(original_roi_data).to_csv(roi_traces_path)
        logger.info(f"Saved original ROI traces to {roi_traces_path}")
        
        # Save background traces
        bg_traces_path = os.path.join(traces_dir, "3_background_traces.csv")
        pd.DataFrame(bg_traces).to_csv(bg_traces_path)
        logger.info(f"Saved background traces to {bg_traces_path}")
    
    # Check if we need to apply slope correction
    correct_slope = bg_config.get("correct_slope", False)
    
    if correct_slope:
        # Get slope correction parameters
        slope_window = bg_config.get("slope_window", [0, 200])  # Default to first 200 frames
        slope_window = [max(0, slope_window[0]), min(n_frames-1, slope_window[1])]
        
        # For storing slope-corrected traces
        slope_corrected_roi = np.zeros_like(roi_data)
        slope_corrected_bg = np.zeros_like(bg_traces)
        
        logger.info(f"Applying slope correction to ROI and background traces (frames {slope_window[0]}-{slope_window[1]})")
        
        for i in range(n_rois):
            # Correct ROI trace
            roi_trace = original_roi_data[i]
            roi_window = roi_trace[slope_window[0]:slope_window[1]+1]
            roi_x = np.arange(len(roi_window))
            
            # Create mask to exclude outliers/peaks in slope estimation
            # Find ROI peaks to exclude from baseline fitting
            from scipy.signal import find_peaks
            roi_peaks, _ = find_peaks(roi_window, prominence=bg_config.get("peak_prominence", 0.05))
            
            # Create mask excluding peaks and their surrounding frames
            mask = np.ones(len(roi_window), dtype=bool)
            for peak in roi_peaks:
                start = max(0, peak - 2)
                end = min(len(roi_window), peak + 3)
                mask[start:end] = False
            
            # If all frames would be excluded, keep at least 50%
            if not np.any(mask) and len(roi_window) > 0:
                logger.warning(f"All frames would be excluded for ROI {i+1}. Using 50% of frames.")
                mask = np.ones(len(roi_window), dtype=bool)
                if len(roi_peaks) > 0:
                    mask[roi_peaks] = False  # Just exclude the exact peaks
            
            # Fit line to non-peak frames
            if np.sum(mask) > 1:
                x_fit = roi_x[mask]
                y_fit = roi_window[mask]
                
                # Use polyfit for linear regression: y = mx + b
                roi_m, roi_b = np.polyfit(x_fit, y_fit, 1)
                
                # Calculate the trend using the estimated slope and intercept
                roi_trend = roi_m * np.arange(n_frames) + roi_b
                
                # Calculate the mean of the baseline points used for fitting
                roi_baseline_mean = np.mean(roi_window[mask])
                
                # Create a flat baseline at the mean level
                roi_flat_baseline = np.ones(n_frames) * roi_baseline_mean
                
                # Replace the trended baseline with a flat baseline
                slope_corrected_roi[i] = roi_trace - roi_trend + roi_flat_baseline
                
                # Log slope information if significant
                if abs(roi_m) > 1e-4:
                    logger.info(f"ROI {i+1} slope: {roi_m:.6f}")
            else:
                logger.warning(f"Not enough non-peak points for ROI {i+1}. Using original trace.")
                slope_corrected_roi[i] = roi_trace
            
            # Correct background trace
            bg_trace = bg_traces[i]
            bg_window = bg_trace[slope_window[0]:slope_window[1]+1]
            bg_x = np.arange(len(bg_window))
            
            # No need to exclude peaks for background, just fit the line
            bg_m, bg_b = np.polyfit(bg_x, bg_window, 1)
            
            # Calculate the trend
            bg_trend = bg_m * np.arange(n_frames) + bg_b
            
            # Calculate the mean
            bg_baseline_mean = np.mean(bg_window)
            
            # Create flat baseline
            bg_flat_baseline = np.ones(n_frames) * bg_baseline_mean
            
            # Replace trend with flat baseline
            slope_corrected_bg[i] = bg_trace - bg_trend + bg_flat_baseline
            
            # Log slope information if significant
            if abs(bg_m) > 1e-4:
                logger.info(f"Background {i+1} slope: {bg_m:.6f}")
        
        # Use the slope-corrected traces for subtraction
        corrected_roi_data = original_roi_data = slope_corrected_roi
        bg_traces = slope_corrected_bg
        
        # Save slope-corrected traces
        if save_intermediate and output_dir:
            # Save slope-corrected ROI traces
            roi_slope_path = os.path.join(traces_dir, "3_roi_traces_slope_corrected.csv")
            pd.DataFrame(slope_corrected_roi).to_csv(roi_slope_path)
            logger.info(f"Saved slope-corrected ROI traces to {roi_slope_path}")
            
            # Save slope-corrected background traces
            bg_slope_path = os.path.join(traces_dir, "3_background_traces_slope_corrected.csv")
            pd.DataFrame(slope_corrected_bg).to_csv(bg_slope_path)
            logger.info(f"Saved slope-corrected background traces to {bg_slope_path}")
    
    # Subtract background from ROI traces
    bg_subtracted = original_roi_data - bg_traces
    
    # Save background-subtracted traces
    if save_intermediate and output_dir:
        bg_subtracted_path = os.path.join(traces_dir, "4_after_background_subtraction.csv")
        pd.DataFrame(bg_subtracted).to_csv(bg_subtracted_path)
        logger.info(f"Saved background-subtracted traces to {bg_subtracted_path}")
    
    logger.info(f"Background subtraction completed in {time.time() - start_time:.2f} seconds")
    return bg_subtracted

def subtract_global_background(
    image_data,
    roi_data,
    roi_masks,
    bg_config,
    logger,
    output_dir=None
):
    """
    Apply global background subtraction using overall image statistics.
    This is especially useful when individual ROI backgrounds are noisy.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Preprocessed image data (frames, height, width)
    roi_data : numpy.ndarray
        ROI fluorescence traces (n_rois, n_frames)
    roi_masks : list of numpy.ndarray
        Binary masks for each ROI
    bg_config : dict
        Background subtraction configuration
    logger : logging.Logger
        Logger object
    output_dir : str, optional
        Output directory for saving intermediate traces
        
    Returns
    -------
    numpy.ndarray
        Background-subtracted ROI traces
    """
    import numpy as np
    import pandas as pd
    import os
    import time
    from scipy.signal import medfilt
    
    start_time = time.time()
    logger.info("Starting global background subtraction")
    
    # Get number of ROIs and frames
    n_rois, n_frames = roi_data.shape
    
    # Create array to store original ROI data for later comparison
    original_roi_data = roi_data.copy()
    
    # Parameters for extracting background
    percentile = bg_config.get("percentile", 0.4)
    median_filter_size = bg_config.get("median_filter_size", 5)
    min_background_area = bg_config.get("min_background_area", 500)
    background_dilation = bg_config.get("background_dilation", 10)
    
    # Create combined ROI mask
    height, width = roi_masks[0].shape
    roi_combined = np.zeros((height, width), dtype=bool)
    for mask in roi_masks:
        roi_combined = roi_combined | mask
    
    # Dilate to avoid pixels near ROIs
    if background_dilation > 0:
        from scipy.ndimage import binary_dilation
        element = np.ones((background_dilation*2+1, background_dilation*2+1), dtype=bool)
        dilated = binary_dilation(roi_combined, structure=element)
    else:
        dilated = roi_combined
    
    # Invert to get background mask
    bg_mask = ~dilated
    
    # Check if we have enough background pixels
    if np.sum(bg_mask) < min_background_area:
        logger.warning(f"Background area too small ({np.sum(bg_mask)} pixels). Using smaller dilation.")
        # Try with smaller dilation
        for dilation_size in [5, 2, 0]:
            if dilation_size > 0:
                element = np.ones((dilation_size*2+1, dilation_size*2+1), dtype=bool)
                dilated = binary_dilation(roi_combined, structure=element)
            else:
                dilated = roi_combined
            bg_mask = ~dilated
            if np.sum(bg_mask) >= min_background_area:
                logger.info(f"Using dilation size {dilation_size} with {np.sum(bg_mask)} background pixels")
                break
    
    # Extract global background trace
    logger.info(f"Extracting global background using {percentile} percentile from {np.sum(bg_mask)} pixels")
    
    # Calculate background for each frame
    bg_values = np.zeros(n_frames)
    for f in range(n_frames):
        frame = image_data[f]
        bg_pixels = frame[bg_mask]
        if len(bg_pixels) > 0:
            bg_values[f] = np.percentile(bg_pixels, percentile * 100)
        else:
            logger.warning(f"No background pixels in frame {f}")
            bg_values[f] = bg_values[f-1] if f > 0 else 0
    
    # Apply median filter to reduce noise
    if median_filter_size > 0:
        bg_values = medfilt(bg_values, median_filter_size)
    
    # Create background trace array (same for all ROIs)
    bg_traces = np.zeros((n_rois, n_frames))
    for i in range(n_rois):
        bg_traces[i] = bg_values
    
    # Save traces before background subtraction
    save_intermediate = bg_config.get("save_intermediate_traces", False)
    if save_intermediate and output_dir:
        traces_dir = os.path.join(output_dir, "intermediate_traces")
        os.makedirs(traces_dir, exist_ok=True)
        
        # Save original traces
        roi_traces_path = os.path.join(traces_dir, "3_roi_traces_before_bg.csv")
        pd.DataFrame(original_roi_data).to_csv(roi_traces_path)
        logger.info(f"Saved original ROI traces to {roi_traces_path}")
        
        # Save background traces
        bg_traces_path = os.path.join(traces_dir, "3_background_traces.csv")
        pd.DataFrame(bg_traces).to_csv(bg_traces_path)
        logger.info(f"Saved background traces to {bg_traces_path}")
    
    # Check if we need to apply slope correction
    correct_slope = bg_config.get("correct_slope", False)
    
    if correct_slope:
        # Get slope correction parameters
        slope_window = bg_config.get("slope_window", [0, 200])  # Default to first 200 frames
        slope_window = [max(0, slope_window[0]), min(n_frames-1, slope_window[1])]
        
        # For storing slope-corrected traces
        slope_corrected_roi = np.zeros_like(roi_data)
        slope_corrected_bg = np.zeros_like(bg_traces)
        
        logger.info(f"Applying slope correction to ROI and background traces (frames {slope_window[0]}-{slope_window[1]})")
        
        for i in range(n_rois):
            # Correct ROI trace
            roi_trace = original_roi_data[i]
            roi_window = roi_trace[slope_window[0]:slope_window[1]+1]
            roi_x = np.arange(len(roi_window))
            
            # Create mask to exclude outliers/peaks in slope estimation
            from scipy.signal import find_peaks
            roi_peaks, _ = find_peaks(roi_window, prominence=bg_config.get("peak_prominence", 0.05))
            
            # Create mask excluding peaks and their surrounding frames
            mask = np.ones(len(roi_window), dtype=bool)
            for peak in roi_peaks:
                start = max(0, peak - 2)
                end = min(len(roi_window), peak + 3)
                mask[start:end] = False
            
            # If all frames would be excluded, keep at least 50%
            if not np.any(mask) and len(roi_window) > 0:
                logger.warning(f"All frames would be excluded for ROI {i+1}. Using 50% of frames.")
                mask = np.ones(len(roi_window), dtype=bool)
                if len(roi_peaks) > 0:
                    mask[roi_peaks] = False  # Just exclude the exact peaks
            
            # Fit line to non-peak frames
            if np.sum(mask) > 1:
                x_fit = roi_x[mask]
                y_fit = roi_window[mask]
                
                # Use polyfit for linear regression: y = mx + b
                roi_m, roi_b = np.polyfit(x_fit, y_fit, 1)
                
                # Calculate the trend using the estimated slope and intercept
                roi_trend = roi_m * np.arange(n_frames) + roi_b
                
                # Calculate the mean of the baseline points used for fitting
                roi_baseline_mean = np.mean(roi_window[mask])
                
                # Create a flat baseline at the mean level
                roi_flat_baseline = np.ones(n_frames) * roi_baseline_mean
                
                # Replace the trended baseline with a flat baseline
                slope_corrected_roi[i] = roi_trace - roi_trend + roi_flat_baseline
                
                # Log slope information if significant
                if abs(roi_m) > 1e-4:
                    logger.info(f"ROI {i+1} slope: {roi_m:.6f}")
            else:
                logger.warning(f"Not enough non-peak points for ROI {i+1}. Using original trace.")
                slope_corrected_roi[i] = roi_trace
        
        # Correct global background trace (only need to do this once)
        bg_window = bg_values[slope_window[0]:slope_window[1]+1]
        bg_x = np.arange(len(bg_window))
        
        # No need to exclude peaks for background, just fit the line
        bg_m, bg_b = np.polyfit(bg_x, bg_window, 1)
        
        # Calculate the trend
        bg_trend = bg_m * np.arange(n_frames) + bg_b
        
        # Calculate the mean
        bg_baseline_mean = np.mean(bg_window)
        
        # Create flat baseline
        bg_flat_baseline = np.ones(n_frames) * bg_baseline_mean
        
        # Replace trend with flat baseline
        corrected_bg_values = bg_values - bg_trend + bg_flat_baseline
        
        # Log slope information if significant
        if abs(bg_m) > 1e-4:
            logger.info(f"Global background slope: {bg_m:.6f}")
        
        # Copy to all ROIs
        for i in range(n_rois):
            slope_corrected_bg[i] = corrected_bg_values
        
        # Use the slope-corrected traces for subtraction
        original_roi_data = slope_corrected_roi
        bg_traces = slope_corrected_bg
        
        # Save slope-corrected traces
        if save_intermediate and output_dir:
            # Save slope-corrected ROI traces
            roi_slope_path = os.path.join(traces_dir, "3_roi_traces_slope_corrected.csv")
            pd.DataFrame(slope_corrected_roi).to_csv(roi_slope_path)
            logger.info(f"Saved slope-corrected ROI traces to {roi_slope_path}")
            
            # Save slope-corrected background traces
            bg_slope_path = os.path.join(traces_dir, "3_background_traces_slope_corrected.csv")
            pd.DataFrame(slope_corrected_bg).to_csv(bg_slope_path)
            logger.info(f"Saved slope-corrected background traces to {bg_slope_path}")
    
    # Subtract background from ROI traces
    bg_subtracted = original_roi_data - bg_traces
    
    # Save background-subtracted traces
    if save_intermediate and output_dir:
        bg_subtracted_path = os.path.join(traces_dir, "4_after_background_subtraction.csv")
        pd.DataFrame(bg_subtracted).to_csv(bg_subtracted_path)
        logger.info(f"Saved background-subtracted traces to {bg_subtracted_path}")
    
    logger.info(f"Global background subtraction completed in {time.time() - start_time:.2f} seconds")
    return bg_subtracted

def extract_roi_fluorescence_with_cnmf(
    roi_path, 
    image_data,
    corrected_data, 
    image_shape, 
    output_dir, 
    config, 
    cnmf_components=None,
    logger=None
):
    """
    Extract ROI fluorescence traces with CNMF refinement.
    
    Parameters
    ----------
    roi_path : str
        Path to the .zip ROI file
    image_data : numpy.ndarray
        Original image data with shape (frames, height, width)
    corrected_data : numpy.ndarray
        CNMF-corrected image data with shape (frames, height, width)
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions
    output_dir : str
        Directory to save ROI masks and visualizations
    config : dict
        Configuration parameters
    cnmf_components : dict, optional
        Dictionary of CNMF components (A: spatial, C: temporal)
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    tuple
        (roi_masks, original_traces, refined_traces) - ROI masks and extracted fluorescence traces
    """
    start_time = time.time()
    
    if logger:
        logger.info(f"Extracting ROIs from {Path(roi_path).name} with CNMF refinement")
    
    # Extract ROIs from zip file
    roi_masks, roi_centers = extract_rois_from_zip(roi_path, image_shape, logger)
    
    # If no valid ROIs were found, log error and return empty results
    if not roi_masks:
        error_msg = f"No valid ROIs were found in {Path(roi_path).name}. Skipping ROI extraction."
        logger.error(error_msg)
        return [], None, None
    
    # Save ROI masks as a reference PNG
    save_roi_visualization(roi_masks, image_shape, output_dir, logger)
    
    # Save masks for CNMF
    save_masks_for_cnmf(roi_masks, output_dir, logger)
    
    # Extract standard fluorescence traces from original data
    original_traces = None
    if image_data is not None:
        logger.info(f"Extracting standard fluorescence traces for {len(roi_masks)} ROIs")
        original_traces = extract_traces(image_data, roi_masks, logger)
    
    # Extract refined traces from corrected data
    refined_traces = None
    if corrected_data is not None:
        logger.info(f"Extracting refined fluorescence traces for {len(roi_masks)} ROIs")
        refined_traces = extract_traces(corrected_data, roi_masks, logger)
    
    logger.info(f"ROI extraction with CNMF refinement completed in {time.time() - start_time:.2f} seconds")
    
    return roi_masks, original_traces, refined_traces

def refine_rois_with_cnmfe(
    roi_path,
    image_data,
    output_dir,
    config,
    logger
):
    """
    Simplified placeholder for ROI refinement with CNMF-E.
    This is a stub function to ensure compatibility with your pipeline.
    
    Parameters
    ----------
    roi_path : str
        Path to the .zip ROI file with manual ROIs
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    output_dir : str
        Directory to save results
    config : dict
        Configuration parameters
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    tuple
        (refined_masks, refined_traces, cnmf_components) - Refined ROI masks, temporal traces, and CNMF components
    """
    logger.info("ROI refinement with CNMF-E is disabled in this version")
    return None, None, None

def refine_rois_with_pnr(
    roi_data,
    roi_masks,
    config,
    logger
):
    """
    Refine ROIs based on peak-to-noise ratio using frequency separation.
    
    Parameters
    ----------
    roi_data : numpy.ndarray
        Fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks (boolean or numeric arrays)
    config : dict
        Configuration parameters for PNR refinement
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    tuple
        (refined_masks, refined_traces, pnr_values) - Refined ROI masks, traces, and PNR values
    """
    import numpy as np
    from scipy import signal
    import time
    import random
    
    start_time = time.time()
    logger.info("Starting ROI refinement based on peak-to-noise ratio")
    
    # Get parameters from config
    noise_freq_cutoff = config.get("pnr_refinement", {}).get("noise_freq_cutoff", 0.03)
    min_pnr = config.get("pnr_refinement", {}).get("min_pnr", 10)
    percentile_threshold = config.get("pnr_refinement", {}).get("percentile_threshold", 99)
    trace_smoothing = config.get("pnr_refinement", {}).get("trace_smoothing", 3)
    auto_determine = config.get("pnr_refinement", {}).get("auto_determine", False)
    
    n_rois, n_frames = roi_data.shape
    logger.info(f"Processing {n_rois} ROIs with {n_frames} timepoints")
    
    # If auto-determination of cutoff frequency is enabled, find the optimal cutoff
    if auto_determine:
        logger.info("Auto-determining optimal frequency cutoff")
        # Sample a subset of ROIs for frequency determination
        n_samples = min(6, n_rois)
        sample_indices = random.sample(range(n_rois), n_samples)
        sample_traces = roi_data[sample_indices]
        
        # Test different frequency values
        noise_freq_list = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]
        
        # Dictionary to store separation quality metrics
        separation_quality = {}
        
        for freq in noise_freq_list:
            # Split into signal and noise components for each frequency
            signal_traces, noise_traces = split_signal_noise(sample_traces, freq, logger)
            
            # Calculate separation quality metric (signal-to-noise ratio)
            signal_power = np.mean(np.var(signal_traces, axis=1))
            noise_power = np.mean(np.var(noise_traces, axis=1))
            snr = signal_power / max(noise_power, 1e-10)
            
            separation_quality[freq] = snr
            logger.info(f"Frequency {freq}: SNR = {snr:.4f}")
        
        # Find frequency with highest separation quality
        optimal_freq = max(separation_quality, key=separation_quality.get)
        logger.info(f"Selected optimal frequency cutoff: {optimal_freq}")
        noise_freq_cutoff = optimal_freq
    
    # Split signal and noise components using the determined cutoff frequency
    logger.info(f"Applying frequency separation with cutoff: {noise_freq_cutoff}")
    signal_traces, noise_traces = split_signal_noise(roi_data, noise_freq_cutoff, logger)
    
    # Compute peak-to-noise ratio for each ROI
    logger.info("Computing peak-to-noise ratio for each ROI")
    pnr_values = compute_pnr(signal_traces, noise_traces, percentile_threshold, logger)
    
    # Apply smoothing to signal traces if requested
    if trace_smoothing > 0:
        logger.info(f"Applying trace smoothing with window size {trace_smoothing}")
        for i in range(n_rois):
            # Use a simple moving average for smoothing
            signal_traces[i] = smooth_trace(signal_traces[i], trace_smoothing)
    
    # Filter ROIs based on PNR threshold
    logger.info(f"Filtering ROIs with PNR threshold: {min_pnr}")
    
    # Find ROIs that meet the threshold
    keep_indices = np.where(pnr_values >= min_pnr)[0]
    n_kept = len(keep_indices)
    
    logger.info(f"Retained {n_kept}/{n_rois} ROIs with PNR >= {min_pnr}")
    
    # Create refined arrays
    if n_kept == 0:
        logger.warning("No ROIs passed the PNR threshold, keeping originals")
        return roi_masks, roi_data, pnr_values
    
    refined_masks = [roi_masks[i] for i in keep_indices]
    refined_traces = signal_traces[keep_indices]  # Use the filtered signal traces
    filtered_pnr = pnr_values[keep_indices]
    
    # Save PNR values for diagnostics
    diagnostic_info = {
        "all_pnr_values": pnr_values.tolist(),
        "kept_indices": keep_indices.tolist(),
        "noise_freq_cutoff": noise_freq_cutoff,
        "min_pnr": min_pnr,
        "percentile_threshold": percentile_threshold
    }
    
    # Log timing and results
    processing_time = time.time() - start_time
    logger.info(f"PNR-based refinement completed in {processing_time:.2f} seconds")
    logger.info(f"Retained {n_kept}/{n_rois} ROIs. PNR range: {np.min(filtered_pnr):.2f} - {np.max(filtered_pnr):.2f}")
    
    return refined_masks, refined_traces, filtered_pnr, diagnostic_info

def split_signal_noise(traces, cutoff_freq, logger=None):
    """
    Split traces into signal and noise components using frequency filtering.
    
    Parameters
    ----------
    traces : numpy.ndarray
        ROI traces with shape (n_rois, n_frames)
    cutoff_freq : float
        Cutoff frequency as fraction of sampling rate (0-0.5)
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    tuple
        (signal_traces, noise_traces) - Separated signal and noise components
    """
    import numpy as np
    from scipy import signal
    
    n_rois, n_frames = traces.shape
    
    # Verify cutoff frequency is in valid range
    if cutoff_freq <= 0 or cutoff_freq >= 0.5:
        if logger:
            logger.warning(f"Invalid cutoff frequency {cutoff_freq}, using default 0.03")
        cutoff_freq = 0.03
    
    # Design Butterworth low-pass filter
    b_low, a_low = signal.butter(2, cutoff_freq, 'low')
    
    # Design Butterworth high-pass filter (same cutoff)
    b_high, a_high = signal.butter(2, cutoff_freq, 'high')
    
    # Initialize output arrays
    signal_traces = np.zeros_like(traces)
    noise_traces = np.zeros_like(traces)
    
    # Apply filters to each ROI
    for i in range(n_rois):
        # Apply low-pass filter for signal
        signal_traces[i] = signal.filtfilt(b_low, a_low, traces[i])
        
        # Apply high-pass filter for noise
        noise_traces[i] = signal.filtfilt(b_high, a_high, traces[i])
        
        # Verify recovery (signal + noise should approximate original)
        recovery_error = np.mean(np.abs(traces[i] - (signal_traces[i] + noise_traces[i])))
        if recovery_error > 1e-10 and logger:
            logger.warning(f"ROI {i+1}: Signal decomposition error: {recovery_error:.6f}")
    
    return signal_traces, noise_traces

def compute_pnr(signal_traces, noise_traces, percentile_threshold=99, logger=None):
    """
    Compute peak-to-noise ratio for each ROI.
    
    Parameters
    ----------
    signal_traces : numpy.ndarray
        Signal component of ROI traces
    noise_traces : numpy.ndarray
        Noise component of ROI traces
    percentile_threshold : float
        Percentile to use for peak detection (default: 99)
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Peak-to-noise ratio for each ROI
    """
    import numpy as np
    
    n_rois = signal_traces.shape[0]
    pnr_values = np.zeros(n_rois)
    
    for i in range(n_rois):
        # Get peak value (using percentile to avoid outliers)
        peak_value = np.percentile(signal_traces[i], percentile_threshold)
        
        # Calculate noise standard deviation
        noise_std = np.std(noise_traces[i])
        
        # Avoid division by zero
        if noise_std > 0:
            pnr_values[i] = peak_value / noise_std
        else:
            pnr_values[i] = 0
            if logger:
                logger.warning(f"ROI {i+1}: Zero noise standard deviation")
    
    return pnr_values

def smooth_trace(trace, window_size):
    """
    Apply moving average smoothing to a trace.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Input trace to smooth
    window_size : int
        Size of smoothing window
        
    Returns
    -------
    numpy.ndarray
        Smoothed trace
    """
    import numpy as np
    from scipy import signal
    
    # Create window coefficients (simple moving average)
    window = np.ones(window_size) / window_size
    
    # Apply convolution for smoothing
    smoothed = signal.convolve(trace, window, mode='same')
    
    # Handle edge effects by copying original values at edges
    half_window = window_size // 2
    if half_window > 0:
        smoothed[:half_window] = trace[:half_window]
        smoothed[-half_window:] = trace[-half_window:]
    
    return smoothed

def visualize_pnr_results(original_traces, signal_traces, noise_traces, pnr_values, indices=None, output_path=None):
    """
    Create visualization of signal-noise separation and PNR values.
    
    Parameters
    ----------
    original_traces : numpy.ndarray
        Original ROI traces
    signal_traces : numpy.ndarray
        Signal component of traces
    noise_traces : numpy.ndarray
        Noise component of traces
    pnr_values : numpy.ndarray
        Peak-to-noise ratio values
    indices : list, optional
        Indices of ROIs to visualize (default: 5 random ROIs)
    output_path : str, optional
        Path to save visualization
        
    Returns
    -------
    dict
        Plot objects if matplotlib is available
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import random
        
        n_rois = original_traces.shape[0]
        
        # Default to 5 random ROIs if indices not provided
        if indices is None:
            n_vis = min(5, n_rois)
            indices = random.sample(range(n_rois), n_vis)
        else:
            n_vis = len(indices)
        
        # Create figure
        fig = plt.figure(figsize=(12, 3 * n_vis))
        
        for i, idx in enumerate(indices):
            # Create original trace plot
            ax1 = fig.add_subplot(n_vis, 3, i*3 + 1)
            ax1.plot(original_traces[idx], 'k-', label=f'Original (ROI {idx+1})')
            ax1.set_title(f'ROI {idx+1} - Original')
            ax1.set_ylabel('Fluorescence')
            
            # Create signal plot
            ax2 = fig.add_subplot(n_vis, 3, i*3 + 2, sharey=ax1)
            ax2.plot(signal_traces[idx], 'g-', label='Signal')
            ax2.set_title(f'Signal Component (PNR: {pnr_values[idx]:.2f})')
            
            # Create noise plot
            ax3 = fig.add_subplot(n_vis, 3, i*3 + 3, sharey=ax1)
            ax3.plot(noise_traces[idx], 'r-', label='Noise')
            ax3.set_title('Noise Component')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return None
        
        return {"figure": fig}
    
    except ImportError:
        print("Matplotlib not available for visualization")
        return None