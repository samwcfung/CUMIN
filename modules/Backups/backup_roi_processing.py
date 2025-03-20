#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Processing Module
------------------
Handles ROI extraction and background subtraction using polygon-based masks.
"""

import os
import time
import random
import numpy as np
import h5py
import cv2
import tifffile
import zipfile
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion, median_filter
from io import BytesIO
from skimage.draw import polygon

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
    
    Parameters
    ----------
    roi_path : str
        Path to the .zip ROI file
    image_data : numpy.ndarray
        Corrected image data with shape (frames, height, width)
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions
    output_dir : str
        Directory to save ROI masks and visualizations
    config : dict
        Configuration parameters
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    tuple
        (roi_masks, roi_fluorescence) - ROI masks and extracted fluorescence traces
    """
    start_time = time.time()
    logger.info(f"Extracting ROIs from {Path(roi_path).name}")
    
    # Extract ROIs from zip file
    roi_masks, roi_centers = extract_rois_from_zip(roi_path, image_shape, logger)
    
    # If no valid ROIs were found, create a placeholder
    if not roi_masks:
        logger.warning("No valid ROIs were found. Creating a placeholder ROI for processing.")
        # Create a small square placeholder ROI in the center of the image
        placeholder_mask = np.zeros(image_shape, dtype=bool)
        center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
        size = min(20, image_shape[0] // 10, image_shape[1] // 10)  # Ensure it fits in the image
        
        # Create a placeholder polygon
        y = [center_y - size, center_y - size, center_y + size, center_y + size]
        x = [center_x - size, center_x + size, center_x + size, center_x - size]
        
        # Use polygon to fill the mask
        rr, cc = polygon(y, x, image_shape)
        placeholder_mask[rr, cc] = True
        
        roi_masks = [placeholder_mask]
        roi_centers = [(center_y, center_x)]
        
        logger.info(f"Created placeholder ROI at center ({center_y}, {center_x}) with size {size*2}x{size*2}")
    
    # Save ROI masks as a reference PNG
    save_roi_visualization(roi_masks, image_shape, output_dir, logger)
    
    # Extract fluorescence traces
    if image_data is not None:
        logger.info(f"Extracting fluorescence traces for {len(roi_masks)} ROIs")
        roi_fluorescence = extract_traces(image_data, roi_masks, logger)
    else:
        logger.warning("No image data provided, skipping fluorescence extraction")
        roi_fluorescence = None
    
    logger.info(f"ROI extraction completed in {time.time() - start_time:.2f} seconds")
    
    return roi_masks, roi_fluorescence

def extract_rois_from_zip(roi_path, image_shape, logger):
    """
    Extract ROI masks from ImageJ/FIJI .zip ROI file using polygon-based approach.
    
    Parameters
    ----------
    roi_path : str
        Path to the .zip ROI file
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    tuple
        (roi_masks, roi_centers) - List of ROI masks and their center coordinates
    """
    roi_masks = []
    roi_centers = []
    
    try:
        # First try to import ijroi
        try:
            import ijroi
            logger.info("Using ijroi library for ROI extraction")
        except ImportError:
            logger.warning("ijroi library not found, trying to use read_roi package")
            try:
                import read_roi
                logger.info("Using read_roi package for ROI extraction")
            except ImportError:
                logger.error("Neither ijroi nor read_roi package found. ROI extraction may fail.")
        
        # Try different ROI reading approaches
        
        # First try using read_roi package which works with zip files directly
        try:
            import read_roi
            logger.info("Attempting to read ROIs with read_roi package")
            
            # Read all ROIs from the zip file
            roi_dict = read_roi.read_roi_zip(roi_path)
            
            if roi_dict:
                logger.info(f"Found {len(roi_dict)} ROIs using read_roi package")
                
                for name, roi in roi_dict.items():
                    try:
                        # Create empty mask
                        mask = np.zeros(image_shape, dtype=bool)
                        
                        # Get coordinates based on ROI type
                        if 'x' in roi and 'y' in roi:
                            # For polygon/freehand ROIs
                            x = np.array(roi['x'])
                            y = np.array(roi['y'])
                            
                            # Ensure coordinates are within image bounds
                            x = np.clip(x, 0, image_shape[1]-1)
                            y = np.clip(y, 0, image_shape[0]-1)
                            
                            # Use skimage's polygon function to create mask
                            rr, cc = polygon(y, x, image_shape)
                            mask[rr, cc] = True
                            
                        elif 'left' in roi and 'top' in roi and 'width' in roi and 'height' in roi:
                            # For rectangle/oval ROIs
                            left = max(0, min(roi['left'], image_shape[1]-1))
                            top = max(0, min(roi['top'], image_shape[0]-1))
                            width = min(roi['width'], image_shape[1] - left)
                            height = min(roi['height'], image_shape[0] - top)
                            
                            if roi.get('type', 'rectangle') == 'oval':
                                # For oval ROIs
                                center = (left + width // 2, top + height // 2)
                                axes = (width // 2, height // 2)
                                cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
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
                            logger.warning(f"Empty ROI mask for {name}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing ROI {name}: {str(e)}")
                
                # If we got any ROIs, return them
                if roi_masks:
                    return roi_masks, roi_centers
                
            else:
                logger.warning("No ROIs found using read_roi package")
                
        except ImportError:
            logger.debug("read_roi package not available")
        except Exception as e:
            logger.warning(f"Error using read_roi package: {str(e)}")
            
        # Try using ijroi approach with BytesIO
        try:
            logger.info("Attempting to read ROIs with ijroi package")
            with zipfile.ZipFile(roi_path, 'r') as zip_ref:
                roi_files = [f for f in zip_ref.namelist() if f.endswith('.roi')]
                logger.info(f"Found {len(roi_files)} ROI files in zip")
                
                for roi_file in roi_files:
                    try:
                        with zip_ref.open(roi_file) as f:
                            roi_data = f.read()
                            
                        # Use BytesIO to create a file-like object for ijroi
                        roi_buffer = BytesIO(roi_data)
                        
                        # Parse ROI using ijroi
                        import ijroi
                        roi = ijroi.read_roi(roi_buffer)
                        
                        # Create empty mask
                        mask = np.zeros(image_shape, dtype=bool)
                        
                        # Handle different ROI types
                        if roi['type'] in ['freehand', 'polygon']:
                            # Get coordinates
                            coords = np.array(roi['coords'])
                            
                            # Ensure coordinates are within image bounds
                            coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[1]-1)
                            coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[0]-1)
                            
                            # Use skimage's polygon function to create mask
                            rr, cc = polygon(coords[:, 1], coords[:, 0], image_shape)
                            mask[rr, cc] = True
                            
                        elif roi['type'] == 'oval':
                            # For oval ROIs
                            left = max(0, min(roi['left'], image_shape[1]-1))
                            top = max(0, min(roi['top'], image_shape[0]-1))
                            width = min(roi['width'], image_shape[1] - left)
                            height = min(roi['height'], image_shape[0] - top)
                            
                            center = (left + width // 2, top + height // 2)
                            axes = (width // 2, height // 2)
                            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
                            
                        elif roi['type'] == 'rect':
                            # For rectangle ROIs
                            left = max(0, min(roi['left'], image_shape[1]-1))
                            top = max(0, min(roi['top'], image_shape[0]-1))
                            width = min(roi['width'], image_shape[1] - left)
                            height = min(roi['height'], image_shape[0] - top)
                            
                            mask[top:top+height, left:left+width] = True
                            
                        else:
                            logger.warning(f"Unsupported ROI type: {roi['type']}")
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
                            logger.warning(f"Empty ROI mask for {roi_file}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing ROI {roi_file} with ijroi: {str(e)}")
        
        except ImportError:
            logger.debug("ijroi package not available")
        except Exception as e:
            logger.warning(f"Error using ijroi approach: {str(e)}")
            
        # Fallback method: Try to directly read ROI data and parse it
        if not roi_masks:
            logger.info("Attempting manual ROI parsing as fallback")
            
            with zipfile.ZipFile(roi_path, 'r') as zip_ref:
                roi_files = [f for f in zip_ref.namelist() if f.endswith('.roi')]
                
                for roi_file in roi_files:
                    try:
                        with zip_ref.open(roi_file) as f:
                            roi_data = f.read()
                        
                        # Check for minimal ROI header 
                        if len(roi_data) < 64:
                            continue
                            
                        # Create mask using basic rectangle ROI parsing
                        top = int.from_bytes(roi_data[8:10], byteorder='big')
                        left = int.from_bytes(roi_data[10:12], byteorder='big')
                        bottom = int.from_bytes(roi_data[12:14], byteorder='big')
                        right = int.from_bytes(roi_data[14:16], byteorder='big')
                        
                        # Ensure coordinates are within image bounds
                        top = max(0, min(top, image_shape[0]-1))
                        left = max(0, min(left, image_shape[1]-1))
                        bottom = max(top+1, min(bottom, image_shape[0]))
                        right = max(left+1, min(right, image_shape[1]))
                        
                        width = right - left
                        height = bottom - top
                        
                        if width <= 0 or height <= 0:
                            continue
                        
                        # Create polygon coordinates for rectangle
                        x = [left, right, right, left]
                        y = [top, top, bottom, bottom]
                        
                        # Create mask
                        mask = np.zeros(image_shape, dtype=bool)
                        rr, cc = polygon(y, x, image_shape)
                        mask[rr, cc] = True
                        
                        # Calculate center
                        center_y = top + height // 2
                        center_x = left + width // 2
                        
                        roi_masks.append(mask)
                        roi_centers.append((center_y, center_x))
                        
                    except Exception as e:
                        logger.warning(f"Manual parsing failed for {roi_file}: {str(e)}")
        
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
    
    Parameters
    ----------
    roi_masks : list
        List of ROI masks
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions
    output_dir : str
        Directory to save the visualization
    logger : logging.Logger
        Logger object
    """
    # Check if we have any masks
    if not roi_masks:
        logger.warning("No ROI masks to visualize")
        
        # Create an empty visualization
        vis_image = np.zeros((*image_shape, 3), dtype=np.uint8)
        
        # Add text indicating no ROIs
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_image, "No ROIs found", (image_shape[1]//4, image_shape[0]//2), 
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save visualization
        vis_path = os.path.join(output_dir, "roi_masks.png")
        cv2.imwrite(vis_path, vis_image)
        
        logger.info(f"Saved empty ROI visualization to {vis_path}")
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
    
    # Save binary masks as a NumPy file for later use
    if roi_masks:
        mask_array = np.stack(roi_masks)
        np_path = os.path.join(output_dir, "roi_masks.npy")
        np.save(np_path, mask_array)
        logger.info(f"Saved binary masks to {np_path}")

def extract_traces(image_data, roi_masks, logger):
    """
    Extract fluorescence traces from image data using ROI masks.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    roi_masks : list
        List of ROI masks
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Array of fluorescence traces with shape (n_rois, n_frames)
    """
    n_frames = image_data.shape[0]
    n_rois = len(roi_masks)
    
    # Array to store fluorescence traces
    traces = np.zeros((n_rois, n_frames))
    
    # Extract fluorescence for each ROI
    for i, mask in enumerate(roi_masks):
        for t in range(n_frames):
            # Extract mean fluorescence within ROI
            if np.any(mask):
                traces[i, t] = np.mean(image_data[t][mask])
            else:
                # Handle empty masks
                logger.warning(f"ROI {i+1} has an empty mask, setting trace to 0")
                traces[i, t] = 0
    
    logger.info(f"Extracted fluorescence traces for {n_rois} ROIs across {n_frames} frames")
    return traces

def subtract_background(
    image_data, 
    roi_traces, 
    roi_masks, 
    bg_config, 
    logger
):
    """
    Subtract background fluorescence from ROI traces.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    roi_traces : numpy.ndarray
        ROI fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks
    bg_config : dict
        Background subtraction configuration
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Background-corrected fluorescence traces
    """
    if roi_traces is None:
        logger.warning("No ROI traces provided, skipping background subtraction")
        return None
    
    start_time = time.time()
    logger.info("Starting background subtraction")
    
    # Select background subtraction method
    method = bg_config.get("method", "darkest_pixels")
    
    if method == "darkest_pixels":
        # Use the darkest pixels in the first frame
        percentile = bg_config.get("percentile", 0.1)
        median_filter_size = bg_config.get("median_filter_size", 3)
        
        # Apply median filter to first frame to reduce noise
        if median_filter_size > 0 and image_data.shape[0] > 0:
            first_frame = median_filter(image_data[0], size=median_filter_size)
        else:
            first_frame = image_data[0] if image_data.shape[0] > 0 else None
        
        if first_frame is None:
            logger.warning("No valid first frame for background calculation, using zero background")
            bg_trace = np.zeros(image_data.shape[0])
        else:
            # Find darkest pixels threshold
            threshold = np.percentile(first_frame, percentile)
            
            # Create background mask
            bg_mask = first_frame < threshold
            
            # Dilate to ensure we're not selecting isolated noise pixels
            dilation_size = bg_config.get("dilation_size", 2)
            if dilation_size > 0:
                bg_mask = binary_dilation(bg_mask, iterations=dilation_size)
            
            # Ensure background mask doesn't overlap with ROIs
            for mask in roi_masks:
                bg_mask[mask] = False
            
            # Calculate background fluorescence over time
            bg_trace = np.zeros(image_data.shape[0])
            for t in range(image_data.shape[0]):
                # Check if we have any background pixels
                if np.any(bg_mask):
                    bg_trace[t] = np.mean(image_data[t][bg_mask])
                else:
                    # Fallback if we have no background pixels
                    logger.warning("No background pixels found, using 0 as background")
                    bg_trace[t] = 0
            
            logger.info(f"Identified {np.sum(bg_mask)} background pixels ({np.sum(bg_mask)/bg_mask.size:.2%} of image)")
        
    elif method == "roi_periphery":
        # Use the periphery of each ROI as local background
        periphery_size = bg_config.get("periphery_size", 2)
        
        # Calculate background for each ROI and subtract
        corrected_traces = np.zeros_like(roi_traces)
        
        for i, mask in enumerate(roi_masks):
            # Create expanded mask
            expanded_mask = binary_dilation(mask, iterations=periphery_size)
            
            # Create periphery mask (expanded - original)
            periphery_mask = expanded_mask & ~mask
            
            # Calculate background for this ROI
            bg_trace_local = np.zeros(image_data.shape[0])
            
            for t in range(image_data.shape[0]):
                # Check if we have any periphery pixels
                if np.any(periphery_mask):
                    bg_trace_local[t] = np.mean(image_data[t][periphery_mask])
                else:
                    bg_trace_local[t] = 0
            
            # Subtract local background
            corrected_traces[i] = roi_traces[i] - bg_trace_local
            
        return corrected_traces
        
    else:
        raise ValueError(f"Unknown background subtraction method: {method}")
    
    # Apply background subtraction
    corrected_traces = roi_traces - bg_trace[np.newaxis, :]
    
    logger.info(f"Background subtraction completed in {time.time() - start_time:.2f} seconds")
    return corrected_traces