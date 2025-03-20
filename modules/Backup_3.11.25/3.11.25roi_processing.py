#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Processing Module
------------------
Handles ROI extraction, refined component extraction from CNMF/CNMF-E,
and background subtraction using polygon-based masks.
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
from scipy import signal
from io import BytesIO
from skimage.draw import polygon
from modules.preprocessing import preprocess_for_cnmfe  # Import missing function

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
    
    # If no valid ROIs were found, log error and return empty results
    if not roi_masks:
        error_msg = f"No valid ROIs were found in {Path(roi_path).name}. Skipping ROI extraction."
        logger.error(error_msg)
        return [], None
    
    # Save ROI masks as a reference PNG
    save_roi_visualization(roi_masks, image_shape, output_dir, logger)
    
    # Save masks in HDF5 and NPY formats for CNMF-E compatibility
    save_masks_for_cnmf(roi_masks, output_dir, logger)
    
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
    Uses a simplified and robust approach for polygon extraction.
    
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
                        logger.debug(f"Processed ROI {i+1} with polygon")
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
                        logger.debug(f"Added ROI {i+1}, center: ({center_x}, {center_y})")
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
                                logger.debug(f"Processed ROI {name} with polygon")
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
                                logger.debug(f"Processed oval ROI: {name}")
                            else:
                                # For rectangle ROIs
                                mask[top:top+height, left:left+width] = True
                                logger.debug(f"Processed rectangle ROI: {name}")
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
                            logger.debug(f"Added ROI {name}, center: ({center_x}, {center_y})")
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
    
    # Save as NPY file (alternative format)
    npy_path = os.path.join(output_dir, "roi_masks.npy")
    np.save(npy_path, mask_array.astype(np.float32))
    logger.info(f"Saved {len(roi_masks)} masks to {npy_path}")
    
    return h5_path

def extract_traces(image_data, roi_masks, logger=None):
    """
    Extract fluorescence traces from ROIs in an image stack.
    """
    import numpy as np
    from skimage.transform import resize
    
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

def get_cnmf_component_masks(cnmf_components, image_shape, threshold=0.1, logger=None):
    """
    Convert CNMF spatial components to binary masks for comparison with original ROIs.
    
    Parameters
    ----------
    cnmf_components : dict
        Dictionary containing CNMF components ('A' key for spatial components)
    image_shape : tuple
        Tuple of (height, width) for the image
    threshold : float, optional
        Threshold for binarizing components, values below this are set to 0
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    list
        List of binary masks corresponding to CNMF components
    """
    import numpy as np
    
    # Check if we have components
    if cnmf_components is None or 'A' not in cnmf_components:
        if logger:
            logger.warning("No CNMF components provided to convert to masks")
        return []
    
    # Get spatial components
    A = cnmf_components['A']
    
    # Convert to dense if sparse
    if hasattr(A, 'toarray'):
        A_dense = A.toarray()
    else:
        A_dense = A
        
    # Reshape to components × pixels
    if A_dense.shape[0] == np.prod(image_shape):
        # A is pixels × components, transpose to components × pixels
        A_dense = A_dense.T
    
    # Get number of components
    n_components = A_dense.shape[0]
    
    if logger:
        logger.info(f"Converting {n_components} CNMF components to binary masks")
    
    # Create masks
    masks = []
    for i in range(n_components):
        # Get component and reshape to image dimensions
        comp = A_dense[i].reshape(image_shape)
        
        # Normalize to [0, 1]
        if np.max(comp) > 0:
            comp = comp / np.max(comp)
            
        # Threshold
        mask = comp > threshold
        
        # Add to list
        masks.append(mask)
    
    return masks

def compare_roi_with_cnmf(roi_masks, cnmf_components, image_shape, output_dir, logger=None):
    """
    Compare original ROI masks with CNMF-derived components and create visualizations.
    
    Parameters
    ----------
    roi_masks : list
        List of original ROI masks
    cnmf_components : dict
        Dictionary containing CNMF components
    image_shape : tuple
        Tuple of (height, width) for the image
    output_dir : str
        Directory to save visualizations
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    dict
        Dictionary with mapping between original ROIs and CNMF components
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    if not roi_masks or cnmf_components is None:
        if logger:
            logger.warning("No ROIs or CNMF components to compare")
        return {}
    
    # Get CNMF component masks
    cnmf_masks = get_cnmf_component_masks(cnmf_components, image_shape, logger=logger)
    
    if not cnmf_masks:
        if logger:
            logger.warning("Could not extract CNMF component masks for comparison")
        return {}
    
    if logger:
        logger.info(f"Comparing {len(roi_masks)} original ROIs with {len(cnmf_masks)} CNMF components")
    
    # Calculate overlap matrix
    n_rois = len(roi_masks)
    n_cnmf = len(cnmf_masks)
    
    overlap_matrix = np.zeros((n_rois, n_cnmf))
    
    for i, roi_mask in enumerate(roi_masks):
        for j, cnmf_mask in enumerate(cnmf_masks):
            # Calculate Jaccard index (intersection over union)
            intersection = np.logical_and(roi_mask, cnmf_mask).sum()
            union = np.logical_or(roi_mask, cnmf_mask).sum()
            
            if union > 0:
                overlap_matrix[i, j] = intersection / union
    
    # Create mapping from ROIs to best matching CNMF components
    roi_to_cnmf = {}
    
    for i in range(n_rois):
        best_match = np.argmax(overlap_matrix[i])
        best_score = overlap_matrix[i, best_match]
        
        if best_score > 0.1:  # Threshold for considering a match
            roi_to_cnmf[i] = {
                'cnmf_index': int(best_match),
                'overlap_score': float(best_score)
            }
    
    if logger:
        logger.info(f"Found matching CNMF components for {len(roi_to_cnmf)} out of {n_rois} ROIs")
    
    # Create visualization
    try:
        plt.figure(figsize=(12, 10))
        
        # Plot overlap matrix as heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(overlap_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Overlap Score')
        plt.xlabel('CNMF Component')
        plt.ylabel('Original ROI')
        plt.title('ROI-CNMF Overlap Matrix')
        
        # Plot examples of matched pairs
        n_examples = min(5, len(roi_to_cnmf))
        
        if n_examples > 0:
            plt.subplot(1, 2, 2)
            
            # Create a composite image showing examples
            composite = np.zeros((*image_shape, 3))
            example_keys = list(roi_to_cnmf.keys())[:n_examples]
            
            for idx, roi_idx in enumerate(example_keys):
                cnmf_idx = roi_to_cnmf[roi_idx]['cnmf_index']
                
                # Add original ROI in red
                composite[roi_masks[roi_idx], 0] = 1.0
                
                # Add CNMF component in green
                composite[cnmf_masks[cnmf_idx], 1] = 1.0
                
                # Overlap will appear yellow
            
            plt.imshow(composite)
            plt.title(f'Example ROI-CNMF Matches (Red=ROI, Green=CNMF)')
            plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roi_cnmf_comparison.png'), dpi=200)
        plt.close()
        
        if logger:
            logger.info(f"Saved ROI-CNMF comparison visualization to {output_dir}")
            
    except Exception as e:
        if logger:
            logger.warning(f"Could not create ROI-CNMF comparison visualization: {str(e)}")
    
    return roi_to_cnmf

def extract_refined_traces_from_cnmf(image_data, roi_masks, cnmf_components, logger=None):
    """
    Extract refined fluorescence traces using CNMF components.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Original image data with shape (frames, height, width)
    roi_masks : list
        List of ROI masks
    cnmf_components : dict
        Dictionary with CNMF components
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Refined fluorescence traces with shape (n_rois, n_frames)
    """
    import numpy as np
    
    # Check inputs
    if image_data is None or not roi_masks or cnmf_components is None:
        if logger:
            logger.warning("Missing required inputs for extracting refined traces")
        return None
    
    # Setup
    n_frames = image_data.shape[0]
    n_rois = len(roi_masks)
    image_shape = image_data.shape[1:]
    
    # Get spatial and temporal components
    A = cnmf_components.get('A')
    C = cnmf_components.get('C')
    
    if A is None or C is None:
        if logger:
            logger.warning("CNMF components missing A or C matrices")
        return None
    
    # Convert spatial components to masks
    cnmf_masks = get_cnmf_component_masks(cnmf_components, image_shape, logger=logger)
    
    if not cnmf_masks:
        if logger:
            logger.warning("Could not create CNMF component masks")
        return None
    
    # Calculate overlap between original ROIs and CNMF components
    n_cnmf = len(cnmf_masks)
    overlap_matrix = np.zeros((n_rois, n_cnmf))
    
    for i, roi_mask in enumerate(roi_masks):
        for j, cnmf_mask in enumerate(cnmf_masks):
            # Calculate overlap (Jaccard index)
            intersection = np.logical_and(roi_mask, cnmf_mask).sum()
            union = np.logical_or(roi_mask, cnmf_mask).sum()
            
            if union > 0:
                overlap_matrix[i, j] = intersection / union
    
    # Initialize traces array
    refined_traces = np.zeros((n_rois, n_frames))
    
    # For each ROI, find best matching CNMF component(s)
    for i in range(n_rois):
        best_match = np.argmax(overlap_matrix[i])
        best_score = overlap_matrix[i, best_match]
        
        if best_score > 0.1:  # Threshold for considering a match
            # Use temporal trace from best matching component
            refined_traces[i] = C[best_match]
            if logger and i < 5:  # Log only a few to avoid spam
                logger.debug(f"ROI {i+1} matched with CNMF component {best_match+1} (overlap: {best_score:.2f})")
        else:
            # If no good match, extract trace from original mask
            if logger and i < 5:
                logger.debug(f"No good CNMF match for ROI {i+1}, using original mask")
            for t in range(n_frames):
                if np.any(roi_masks[i]):
                    refined_traces[i, t] = np.mean(image_data[t][roi_masks[i]])
                else:
                    refined_traces[i, t] = 0
    
    return refined_traces

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
        (roi_masks, roi_fluorescence, refined_fluorescence) - ROI masks and extracted fluorescence traces
    """
    import time
    from pathlib import Path
    
    start_time = time.time()
    if logger:
        logger.info(f"Extracting ROIs from {Path(roi_path).name} with CNMF refinement")
    
    # Extract ROIs from zip file
    roi_masks, roi_centers = extract_rois_from_zip(roi_path, image_shape, logger)
    
    # If no valid ROIs were found, log error and return empty results
    if not roi_masks:
        error_msg = f"No valid ROIs were found in {Path(roi_path).name}. Skipping ROI extraction."
        if logger:
            logger.error(error_msg)
        return [], None, None
    
    # Save ROI masks as a reference PNG
    save_roi_visualization(roi_masks, image_shape, output_dir, logger)
    
    # Save masks in CNMF-compatible format
    masks_path = save_masks_for_cnmf(roi_masks, output_dir, logger)
    
    # Extract standard fluorescence traces from original data
    if image_data is not None:
        if logger:
            logger.info(f"Extracting standard fluorescence traces for {len(roi_masks)} ROIs")
        roi_fluorescence = extract_traces(image_data, roi_masks, logger)
    else:
        if logger:
            logger.warning("No original image data provided, skipping standard fluorescence extraction")
        roi_fluorescence = None
    
    # Extract refined fluorescence traces from CNMF components
    if corrected_data is not None and cnmf_components is not None:
        if logger:
            logger.info(f"Extracting refined fluorescence traces using CNMF components")
        refined_fluorescence = extract_refined_traces_from_cnmf(corrected_data, roi_masks, cnmf_components, logger)
        
        # Compare original ROIs with CNMF components
        compare_roi_with_cnmf(roi_masks, cnmf_components, image_shape, output_dir, logger)
    elif corrected_data is not None:
        if logger:
            logger.info(f"No CNMF components provided, extracting traces from corrected data")
        refined_fluorescence = extract_traces(corrected_data, roi_masks, logger)
    else:
        if logger:
            logger.warning("No corrected data provided, skipping refined fluorescence extraction")
        refined_fluorescence = None
    
    if logger:
        logger.info(f"ROI extraction with CNMF refinement completed in {time.time() - start_time:.2f} seconds")
    
    return roi_masks, roi_fluorescence, refined_fluorescence

def refine_rois_with_cnmfe(
    roi_path,
    image_data,
    output_dir,
    config,
    logger
):
    """
    Optimized version of ROI refinement using CNMF-E with float16 and downsampling.
    
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
    import time
    from pathlib import Path
    import os
    import shutil
    import tempfile
    import gc
    import numpy as np
    
    start_time = time.time()
    logger.info(f"Starting optimized ROI refinement with CNMF-E for {Path(roi_path).name}")
    
    # Extract optimization parameters
    optimization = config.get("optimization", {})
    spatial_downsample = optimization.get("spatial_downsample", 2)
    use_float16 = optimization.get("use_float16", True)
    simplified_mode = optimization.get("simplified_mode", True)
    
    # Create a unique output directory for temporary files
    temp_dir = os.path.join(output_dir, "temp_caiman_files")
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Import necessary modules
        try:
            import caiman as cm
            from caiman.source_extraction.cnmf import cnmf as cnmf
            from caiman.source_extraction.cnmf import params as params
            from scipy import sparse
        except ImportError as e:
            logger.error(f"Failed to import CaImAn: {str(e)}")
            return None, None, None
        
        # Get image dimensions
        if image_data is None:
            logger.error("No image data provided for CNMF-E refinement")
            return None, None, None
        
        # Store original dimensions before preprocessing
        orig_n_frames, orig_height, orig_width = image_data.shape
        
        # Preprocess data (downsample and convert to float16)
        logger.info("Preprocessing data to optimize memory usage and speed")
        preprocessed_data = preprocess_for_cnmfe(
            image_data,
            spatial_downsample=spatial_downsample,
            temporal_downsample=1,  # Keep all frames for accurate temporal dynamics
            use_float16=use_float16,
            logger=logger
        )
        
        # Get new dimensions after preprocessing
        n_frames, height, width = preprocessed_data.shape
        image_shape = (height, width)
        
        # Extract ROIs from zip file and resize if needed
        roi_masks, roi_centers = extract_rois_from_zip(roi_path, (orig_height, orig_width), logger)
        
        # If no valid ROIs were found, log error and return empty results
        if not roi_masks:
            logger.error(f"No valid ROIs were found in {Path(roi_path).name}")
            return None, None, None
        
        # Downsample ROIs if needed
        if spatial_downsample > 1:
            logger.info(f"Downsampling ROIs by {spatial_downsample}x to match data")
            from skimage.transform import resize
            
            downsampled_masks = []
            for mask in roi_masks:
                # Use nearest neighbor interpolation to maintain binary nature
                downsampled = resize(mask.astype(float), image_shape, 
                                   order=0, preserve_range=True) > 0.5
                downsampled_masks.append(downsampled)
            
            roi_masks = downsampled_masks
        
        # Convert masks to numpy array
        masks_array = np.stack(roi_masks).astype(np.float16 if use_float16 else np.float32)
        
        # Extract CNMF-E parameters from config
        cnmf_e_params = config.get("cnmf_e", {})
        
        # Set default parameters if not specified, adjusting for downsampling
        min_corr = cnmf_e_params.get("min_corr", 0.8)
        min_pnr = cnmf_e_params.get("min_pnr", 10)
        
        # Scale gSig for downsampled data
        orig_gSig = cnmf_e_params.get("gSig", (3, 3))
        gSig = tuple(s/spatial_downsample for s in orig_gSig)
        
        # Scale gSiz accordingly
        gSiz = tuple(2 * np.array(gSig) + 1)
        
        ring_size_factor = cnmf_e_params.get("ring_size_factor", 1.4)
        merge_thresh = cnmf_e_params.get("merge_thresh", 0.7)
        
        # Optimization: Use faster params for simplified mode
        if simplified_mode:
            # Use less rigid parameters for faster processing
            min_corr *= 0.9      # More lenient correlation threshold
            merge_thresh *= 0.9  # More lenient merging
        
        # Number of processes for parallelization
        use_multiprocessing = config.get("use_multiprocessing", False)
        n_processes = config.get("n_processes", 1)
        
        logger.info(f"CNMF-E parameters: gSig={gSig}, min_corr={min_corr}, min_pnr={min_pnr}")
        
        # Create parameters dictionary
        params_dict = {}
        
        # Dataset parameters
        params_dict['fr'] = config.get("frame_rate", 30.0)  # frame rate
        params_dict['decay_time'] = 0.4                     # length of a typical transient (seconds)
        
        # CNMF parameters
        params_dict['p'] = 1                        # order of AR model
        params_dict['nb'] = 0                       # number of background components (0 for CNMF-E)
        params_dict['merge_thresh'] = merge_thresh  # merging threshold
        
        # Don't use patches for downsampled data (much faster)
        if height * width <= 256 * 256:
            # Small enough to process without patches
            params_dict['rf'] = None
            params_dict['stride'] = None
            logger.info("Data small enough to process without patches")
        else:
            # Use patches but with larger stride (faster)
            params_dict['rf'] = 40                  # half-size of patches
            params_dict['stride'] = 30              # larger stride = fewer patches
            logger.info("Using patches with increased stride for faster processing")
        
        # Component detection parameters
        params_dict['K'] = None                     # Auto-detect components per patch
        params_dict['gSig'] = gSig                  # Expected half size of neurons (scaled)
        params_dict['gSiz'] = gSiz                  # Size of bounding box (scaled)
        params_dict['min_corr'] = min_corr          # Min correlation
        params_dict['min_pnr'] = min_pnr            # Min peak-to-noise ratio
        
        # CNMF-E specific parameters
        params_dict['method_init'] = 'corr_pnr'     # Initialization method
        params_dict['ring_size_factor'] = ring_size_factor
        params_dict['center_psf'] = True            # Center the PSF
        
        # Spatial parameters - Simplified for speed
        if simplified_mode:
            params_dict['update_background_components'] = False  # Skip background updates
            params_dict['ssub'] = 1                     # No further spatial downsampling
            params_dict['tsub'] = 4                     # Aggressive temporal downsampling
        else:
            params_dict['update_background_components'] = True
            params_dict['ssub'] = 1
            params_dict['tsub'] = 2
        
        # Initialize parameters
        cnm_params = params.CNMFParams(params_dict=params_dict)
        
        # Convert data to appropriate type for CaImAn
        # CaImAn requires float32 internally for most functions
        Y = preprocessed_data.astype(np.float32)
        
        # Set up multiprocessing
        dview = None
        if use_multiprocessing and n_processes > 1:
            try:
                # Try to use the CaImAn cluster setup
                logger.info(f"Setting up CaImAn cluster with {n_processes} processes")
                _, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=n_processes, single_thread=False
                )
                logger.info(f"Cluster setup complete with {n_processes} processes")
            except Exception as e:
                logger.warning(f"Could not set up CaImAn cluster: {str(e)}")
                dview = None
        
        # Create CNMF object
        cnm = cnmf.CNMF(n_processes=n_processes, params=cnm_params, dview=dview)
        
        # Initialize with the pre-defined ROIs
        logger.info(f"Initializing CNMF-E with {len(roi_masks)} pre-defined ROIs")
        
        # Reshape masks for initialization
        n_masks = masks_array.shape[0]
        masks_flat = masks_array.reshape(n_masks, -1).T  # pixels x components
        
        # Normalize masks
        for i in range(n_masks):
            col_max = np.max(masks_flat[:, i])
            if col_max > 0:
                masks_flat[:, i] = masks_flat[:, i] / col_max
        
        # Reshape masks for initialization
        n_masks = masks_array.shape[0]
        masks_flat = masks_array.reshape(n_masks, -1).T  # pixels x components
        
        # Normalize masks
        for i in range(n_masks):
            col_max = np.max(masks_flat[:, i])
            if col_max > 0:
                masks_flat[:, i] = masks_flat[:, i] / col_max
        
        # Ensure correct dtype for compatibility with CNMF-E
        masks_flat = masks_flat.astype(np.float32)
        for i in range(n_masks):
            col_max = np.max(masks_flat[:, i])
            if col_max > 0:
                masks_flat[:, i] = masks_flat[:, i] / col_max
        
        # Convert to sparse matrix
        from scipy import sparse
        A_init = sparse.csc_matrix(masks_flat)
        
        # Initialize with ROIs
        try:
            if hasattr(cnm, 'estimates'):
                cnm.estimates = cnmf.Estimates(A=A_init, dims=image_shape)
            else:
                cnm.A = A_init
                cnm.dims = image_shape
            
            logger.info("Successfully initialized with pre-defined ROIs")
        except Exception as e:
            logger.warning(f"Could not initialize with ROIs: {str(e)}")
        
        # Fit model
        logger.info("Running CNMF-E fit - this is the time-intensive step")
        fit_start = time.time()
        
        try:
            # Fit with existing initialization
            cnm = cnm.fit(Y)
            logger.info(f"CNMF-E fit completed in {time.time() - fit_start:.2f} seconds")
        except Exception as e:
            logger.error(f"Error in CNMF-E fit: {str(e)}")
            if dview is not None:
                try:
                    cm.stop_server(dview=dview)
                except:
                    pass
            return None, None, None
        
        # Get components
        components = get_cnmf_components(cnm, logger=logger)
        
        # Check for valid components
        if 'A' not in components or 'C' not in components:
            logger.warning("No valid components found")
            return None, None, None
        
        # Get number of components
        n_components = components['A'].shape[1] if hasattr(components['A'], 'shape') else 0
        logger.info(f"Found {n_components} components")
        
        # Get indices of good components
        if simplified_mode:
            # In simplified mode, just use all components
            idx_components = np.arange(n_components)
            logger.info("Using all components (simplified mode)")
        else:
            # Try to evaluate components
            try:
                # Set quality parameters
                cnm.params.set('quality', {'min_SNR': 2.0,
                                         'rval_thr': 0.85,
                                         'use_cnn': False})
                
                # Evaluate components
                cnm.estimates.evaluate_components(Y, cnm.params)
                
                # Update components
                components = get_cnmf_components(cnm, logger=logger)
                idx_components = components.get('idx_components', None)
                
                if idx_components is not None and len(idx_components) > 0:
                    logger.info(f"Component evaluation: {len(idx_components)} accepted")
                else:
                    idx_components = np.arange(n_components)
                    logger.info("No components passed evaluation, using all")
            except Exception as e:
                logger.warning(f"Component evaluation failed: {str(e)}")
                idx_components = np.arange(n_components)
        
        # Convert spatial components to binary masks
        A = components['A']
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        # Extract good components
        A_good = A_dense[:, idx_components]
        
        # Create masks from components
        refined_masks = []
        for i in range(len(idx_components)):
            comp = A_good[:, i].reshape(image_shape)
            # Normalize
            comp_max = np.max(comp)
            if comp_max > 0:
                comp = comp / comp_max
            # Threshold
            mask = comp > 0.2
            refined_masks.append(mask)
        
        # Extract temporal components
        C = components['C']
        refined_traces = np.zeros((len(idx_components), n_frames))
        for i, idx in enumerate(idx_components):
            refined_traces[i] = C[idx]
        
        # Upscale masks if we downsampled
        if spatial_downsample > 1:
            logger.info(f"Upscaling refined masks to original dimensions")
            
            upscaled_masks = []
            for mask in refined_masks:
                # Use nearest neighbor interpolation to maintain binary nature
                upscaled = resize(mask.astype(float), (orig_height, orig_width), 
                                order=0, preserve_range=True) > 0.5
                upscaled_masks.append(upscaled)
            
            refined_masks = upscaled_masks
        
        # Save refined masks
        refined_masks_array = np.stack(refined_masks)
        refined_masks_path = os.path.join(output_dir, "refined_masks.npy")
        np.save(refined_masks_path, refined_masks_array)
        logger.info(f"Saved {len(refined_masks)} refined masks to {refined_masks_path}")
        
        # Save visualization
        refine_output_dir = os.path.join(output_dir, "refined")
        os.makedirs(refine_output_dir, exist_ok=True)
        save_roi_visualization(refined_masks, (orig_height, orig_width), refine_output_dir, logger)
        
        # Create components dictionary
        cnmf_components = {
            'A': A,
            'C': C,
            'b': components.get('b', None),
            'f': components.get('f', None),
            'S': components.get('S', None),
            'idx_components': idx_components,
            'idx_components_bad': components.get('idx_components_bad', None),
            'original_masks': roi_masks,
            'refined_masks': refined_masks,
            'spatial_downsample': spatial_downsample
        }
        
        # Save components
        components_path = os.path.join(output_dir, "cnmf_components.h5")
        try:
            with h5py.File(components_path, 'w') as f:
                # Main group
                comp_group = f.create_group('cnmf_components')
                
                # Spatial components (A)
                comp_group.create_dataset('A', data=A_dense)
                
                # Temporal components (C)
                comp_group.create_dataset('C', data=C)
                
                # Background components if available
                b = components.get('b', None)
                if b is not None:
                    if sparse.issparse(b):
                        b_dense = b.toarray()
                    else:
                        b_dense = b
                    comp_group.create_dataset('b', data=b_dense)
                
                f_comp = components.get('f', None)
                if f_comp is not None:
                    comp_group.create_dataset('f', data=f_comp)
                
                # Store indices
                comp_group.create_dataset('idx_components', data=np.array(idx_components))
                
                # Metadata
                meta_group = f.create_group('metadata')
                meta_group.attrs['n_components'] = n_components
                meta_group.attrs['n_accepted'] = len(idx_components)
                meta_group.attrs['spatial_downsample'] = spatial_downsample
                meta_group.attrs['use_float16'] = use_float16
                meta_group.attrs['simplified_mode'] = simplified_mode
                meta_group.attrs['processing_time'] = time.time() - start_time
                
            logger.info(f"Saved components to {components_path}")
        except Exception as e:
            logger.warning(f"Error saving components: {str(e)}")
        
        # Clean up
        if dview is not None:
            try:
                cm.stop_server(dview=dview)
            except:
                pass
        
        logger.info(f"Optimized ROI refinement completed in {time.time() - start_time:.2f} seconds")
        return refined_masks, refined_traces, cnmf_components
        
    except Exception as e:
        logger.error(f"Error in optimized CNMF-E processing: {str(e)}", exc_info=True)
        return None, None, None
    finally:
        # Clean up temp files
        try:
            # Only attempt to remove if directory exists
            if os.path.exists(temp_dir):
                # Try to list directory contents
                try:
                    files = os.listdir(temp_dir)
                    # Try to delete each file
                    for file in files:
                        try:
                            file_path = os.path.join(temp_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except:
                            pass
                except:
                    pass
                
                # Try to remove the directory itself
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
        except:
            pass

def subtract_background(image_data, roi_data, roi_masks, config, logger=None):
    """
    Subtract background from ROI fluorescence traces.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image stack with shape (frames, height, width)
    roi_data : numpy.ndarray
        ROI fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list of numpy.ndarray
        List of binary masks for each ROI
    config : dict
        Background subtraction configuration
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Background-corrected fluorescence traces
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, median_filter
    from skimage.transform import resize
    from scipy import signal
    
    # Get method and parameters
    method = config.get("method", "roi_periphery")
    periphery_size = config.get("periphery_size", 2)
    percentile = config.get("percentile", 0.1)
    median_filter_size = config.get("median_filter_size", 3)
    dilation_size = config.get("dilation_size", 2)
    cutoff_freq = config.get("cutoff_freq", 0.001)
    filter_order = config.get("filter_order", 2)
    
    if logger:
        logger.info(f"Subtracting background using method: {method}")
    
    # Get dimensions
    n_rois = len(roi_masks)
    n_frames = image_data.shape[0]
    _, img_height, img_width = image_data.shape
    
    # Check if we need to resize the masks
    try:
        mask_height, mask_width = roi_masks[0].shape
        needs_resize = (mask_height != img_height or mask_width != img_width)
    except:
        needs_resize = True
        
    # Resize masks if needed
    if needs_resize:
        if logger:
            logger.warning(f"ROI mask dimensions don't match image dimensions in background subtraction")
            logger.info(f"Resizing ROI masks for background subtraction")
            
        resized_masks = []
        for mask in roi_masks:
            # Convert to numpy array if needed
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=bool)
                
            # Convert to float for resize operation
            mask_float = mask.astype(np.float32)
            
            # Resize using nearest-neighbor interpolation
            resized_mask = resize(mask_float, (img_height, img_width), 
                                 order=0, preserve_range=True, anti_aliasing=False)
                                 
            # Convert back to boolean
            resized_mask = resized_mask > 0.5
            resized_masks.append(resized_mask)
            
        roi_masks = resized_masks
        
        if logger:
            logger.info(f"ROI masks resized for background subtraction")
    
    # Initialize background-corrected traces
    bg_corrected = np.copy(roi_data)
    
    # Apply background subtraction based on method
    if method == "roi_periphery":
        if logger:
            logger.info(f"Using ROI periphery method with periphery size {periphery_size}")
        
        for i, mask in enumerate(roi_masks):
            # Create periphery mask
            expanded_mask = binary_dilation(mask, iterations=periphery_size)
            periphery_mask = expanded_mask & ~mask
            
            # Skip if periphery is empty
            if not np.any(periphery_mask):
                if logger:
                    logger.warning(f"ROI {i+1}: No valid periphery pixels, skipping background subtraction")
                continue
            
            # Extract background traces from periphery
            bg_trace = np.zeros(n_frames)
            for t in range(n_frames):
                bg_values = image_data[t][periphery_mask]
                bg_trace[t] = np.mean(bg_values)
            
            # Subtract background
            bg_corrected[i] = roi_data[i] - bg_trace
            
    elif method == "darkest_pixels":
        if logger:
            logger.info(f"Using darkest pixels method with percentile {percentile}")
        
        # Create global darkest pixels mask
        darkest_pixels_mask = np.zeros((img_height, img_width), dtype=bool)
        
        # Compute average intensity per pixel
        avg_intensity = np.mean(image_data, axis=0)
        
        # Find darkest percentile
        threshold = np.percentile(avg_intensity, percentile * 100)
        darkest_pixels_mask = avg_intensity <= threshold
        
        # Apply median filter to remove noise
        if median_filter_size > 0:
            darkest_pixels_mask = median_filter(darkest_pixels_mask.astype(float), 
                                               size=median_filter_size) > 0.5
        
        # Dilate the mask to get a more robust background region
        if dilation_size > 0:
            darkest_pixels_mask = binary_dilation(darkest_pixels_mask, iterations=dilation_size)
        
        # Extract background trace
        bg_trace = np.zeros(n_frames)
        for t in range(n_frames):
            bg_values = image_data[t][darkest_pixels_mask]
            if len(bg_values) > 0:
                bg_trace[t] = np.mean(bg_values)
        
        # Apply subtraction to all ROIs
        for i in range(n_rois):
            bg_corrected[i] = roi_data[i] - bg_trace
            
    elif method == "cnmf_background":
        if logger:
            logger.info(f"Using CNMF background components (no additional subtraction needed)")
        # CNMF background separation was already done during extraction
        pass
        
    elif method == "lowpass_filter":
        if logger:
            logger.info(f"Using lowpass filter method with cutoff {cutoff_freq}, order {filter_order}")
        
        # Design Butterworth low-pass filter
        b, a = signal.butter(filter_order, cutoff_freq, 'low')
        
        # Apply to each ROI trace
        for i in range(n_rois):
            # Get the ROI trace
            trace = roi_data[i]
            
            # Apply filter to capture slow changes (background)
            bg_trace = signal.filtfilt(b, a, trace)
            
            # Subtract background (slow trend)
            bg_corrected[i] = trace - bg_trace
    
    else:
        if logger:
            logger.warning(f"Unknown background subtraction method: {method}, using original traces")
    
    return bg_corrected

def correct_signal_polarity(traces, baseline_frames, analysis_frames, logger=None):
    """
    Detect and correct inverted fluorescence signals.
    
    Parameters
    ----------
    traces : numpy.ndarray
        ROI traces with shape (n_rois, n_frames)
    baseline_frames : tuple
        (start, end) frames for baseline period
    analysis_frames : tuple
        (start, end) frames for analysis/response period
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Corrected traces with consistent polarity
    """
    corrected_traces = traces.copy()
    n_rois = traces.shape[0]
    
    for i in range(n_rois):
        # Get baseline and response segments
        baseline = traces[i, baseline_frames[0]:baseline_frames[1]]
        response = traces[i, analysis_frames[0]:analysis_frames[1]]
        
        # Calculate mean change from baseline to response
        baseline_mean = np.mean(baseline)
        response_mean = np.mean(response)
        change = response_mean - baseline_mean
        
        # Calculate response magnitude (absolute change)
        mag = np.abs(change)
        
        # If the change is negative and significant, flip the trace
        if change < 0 and mag > 0.2:  # Only flip significant responses
            if logger:
                logger.info(f"Correcting inverted polarity for ROI {i+1}")
            corrected_traces[i] = -traces[i]
    
    return corrected_traces

def add_rois_to_existing_file(roi_path, h5_path, image_shape, logger=None):
    """
    Add ROIs from an ImageJ/FIJI .zip ROI file to an existing HDF5 file.
    
    Parameters
    ----------
    roi_path : str
        Path to the .zip ROI file
    h5_path : str
        Path to the existing HDF5 file
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    import h5py
    
    # Extract ROIs from zip file
    if logger:
        logger.info(f"Extracting ROIs from {Path(roi_path).name}")
    
    roi_masks, roi_centers = extract_rois_from_zip(roi_path, image_shape, logger)
    
    # If no valid ROIs were found, log error and return
    if not roi_masks:
        if logger:
            logger.error(f"No valid ROIs were found in {Path(roi_path).name}. Cannot add to HDF5 file.")
        return False
    
    # Convert list of masks to array
    mask_array = np.stack(roi_masks).astype(np.float32)
    
    try:
        # Open the existing HDF5 file
        with h5py.File(h5_path, 'a') as f:
            # Check if 'masks' dataset already exists
            if 'masks' in f:
                # Get existing masks
                existing_masks = f['masks'][:]
                
                # Combine with new masks
                combined_masks = np.concatenate([existing_masks, mask_array], axis=0)
                
                # Delete existing dataset
                del f['masks']
                
                # Create new dataset
                f.create_dataset('masks', data=combined_masks)
                f.attrs['n_masks'] = combined_masks.shape[0]
                
                if logger:
                    logger.info(f"Added {mask_array.shape[0]} new masks to existing {existing_masks.shape[0]} masks")
            else:
                # Create new dataset
                f.create_dataset('masks', data=mask_array)
                f.attrs['n_masks'] = mask_array.shape[0]
                
                if logger:
                    logger.info(f"Created new masks dataset with {mask_array.shape[0]} masks")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error adding ROIs to HDF5 file: {str(e)}")
        return False