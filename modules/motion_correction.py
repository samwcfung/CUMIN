#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Correction Module
---------------------
Implements motion correction using NoRMCorre algorithm.
"""

import os
import time
import numpy as np
import h5py
import tifffile
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Try to import NoRMCorre
try:
    from normcorre.locally_rigid_registration import register_frame, apply_shifts
    from normcorre.rigid_registration import register_frame as register_frame_rigid
    from normcorre.rigid_registration import apply_shifts as apply_shifts_rigid
    HAS_NORMCORRE = True
except ImportError:
    HAS_NORMCORRE = False
    warnings.warn("NoRMCorre not found. Motion correction will be unavailable. Install with: pip install normcorre")


def apply_normcorre_correction(image_data, params, logger=None):
    """
    Apply NoRMCorre motion correction to image data.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Input data with shape (frames, height, width)
    params : dict
        Parameters for motion correction
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    tuple
        (corrected_data, shifts) - motion-corrected data and calculated shifts
    """
    if not HAS_NORMCORRE:
        if logger:
            logger.error("NoRMCorre not installed. Install with: pip install normcorre")
        raise ImportError("NoRMCorre not installed")
    
    start_time = time.time()
    
    if logger:
        logger.info(f"Starting motion correction with parameters: {params}")
    
    # Extract parameters
    max_shift = params.get('max_shift', 10)
    patch_size = params.get('patch_size', [50, 50])
    method = params.get('method', 'rigid')
    
    # Get dimensions
    n_frames, height, width = image_data.shape
    
    # Convert to float32 if needed
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)
    
    # Initialize corrected data array
    corrected_data = np.zeros_like(image_data)
    
    # Compute template from first few frames
    n_template_frames = min(30, n_frames)
    template = np.mean(image_data[:n_template_frames], axis=0)
    
    # Initialize shifts storage
    if method == 'rigid':
        shifts = np.zeros((n_frames, 2))
    else:
        # For non-rigid, we'll store shifts for each patch
        n_patches_y = int(np.ceil(height / patch_size[0]))
        n_patches_x = int(np.ceil(width / patch_size[1]))
        shifts = np.zeros((n_frames, n_patches_y, n_patches_x, 2))
    
    # Create registration options
    options = {
        'max_shift': max_shift,
        'keep_original': False
    }
    
    if method == 'nonrigid':
        # Add non-rigid specific options
        options['grid_size'] = patch_size
        options['mot_uf'] = 4  # Motion upsampling factor
        options['bin_width'] = 200  # Width of each bin
    
    # Process each frame
    for i in range(n_frames):
        try:
            if method == 'rigid':
                # Rigid registration
                shift, _ = register_frame_rigid(
                    template,
                    image_data[i],
                    options=options
                )
                shifts[i] = shift
                corrected_data[i] = apply_shifts_rigid(image_data[i], shift, options=options)
            else:
                # Non-rigid registration
                shift, _, _ = register_frame(
                    template,
                    image_data[i],
                    options=options
                )
                shifts[i] = shift
                corrected_data[i] = apply_shifts(image_data[i], shift, options=options)
            
            # Log progress periodically
            if logger and i % 10 == 0:
                elapsed = time.time() - start_time
                frames_per_second = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {i+1}/{n_frames} frames ({frames_per_second:.2f} frames/sec)")
                
        except Exception as e:
            if logger:
                logger.warning(f"Error processing frame {i}: {str(e)}")
            # Use uncorrected frame as fallback
            corrected_data[i] = image_data[i]
    
    total_time = time.time() - start_time
    if logger:
        logger.info(f"Motion correction completed in {total_time:.2f} seconds")
        logger.info(f"Average processing speed: {n_frames/total_time:.2f} frames/sec")
    
    return corrected_data, shifts


def estimate_motion(image_data, method='rigid', max_shift=10, patch_size=None, logger=None):
    """
    Estimate motion in image data without applying correction.
    Useful for determining if motion correction is needed.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Input data with shape (frames, height, width)
    method : str, optional
        Motion estimation method ('rigid' or 'nonrigid')
    max_shift : int, optional
        Maximum shift in pixels
    patch_size : list, optional
        Size of patches for non-rigid motion [height, width]
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    numpy.ndarray
        Estimated motion shifts
    """
    if not HAS_NORMCORRE:
        if logger:
            logger.error("NoRMCorre not installed. Install with: pip install normcorre")
        raise ImportError("NoRMCorre not installed")
    
    start_time = time.time()
    
    if logger:
        logger.info(f"Estimating motion with method: {method}")
    
    # Get dimensions
    n_frames, height, width = image_data.shape
    
    # Set default patch size if not provided
    if patch_size is None:
        patch_size = [50, 50]
    
    # Convert to float32 if needed
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)
    
    # Compute template from first few frames
    n_template_frames = min(30, n_frames)
    template = np.mean(image_data[:n_template_frames], axis=0)
    
    # Initialize shifts storage
    if method == 'rigid':
        shifts = np.zeros((n_frames, 2))
    else:
        # For non-rigid, we'll store shifts for each patch
        n_patches_y = int(np.ceil(height / patch_size[0]))
        n_patches_x = int(np.ceil(width / patch_size[1]))
        shifts = np.zeros((n_frames, n_patches_y, n_patches_x, 2))
    
    # Create registration options
    options = {
        'max_shift': max_shift,
        'keep_original': False
    }
    
    if method == 'nonrigid':
        # Add non-rigid specific options
        options['grid_size'] = patch_size
        options['mot_uf'] = 4  # Motion upsampling factor
        options['bin_width'] = 200  # Width of each bin
    
    # Process each frame to estimate motion
    for i in range(n_frames):
        try:
            if method == 'rigid':
                # Rigid registration
                shift, _ = register_frame_rigid(
                    template,
                    image_data[i],
                    options=options
                )
                shifts[i] = shift
            else:
                # Non-rigid registration
                shift, _, _ = register_frame(
                    template,
                    image_data[i],
                    options=options
                )
                shifts[i] = shift
            
            # Log progress periodically
            if logger and i % 10 == 0:
                elapsed = time.time() - start_time
                frames_per_second = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {i+1}/{n_frames} frames ({frames_per_second:.2f} frames/sec)")
                
        except Exception as e:
            if logger:
                logger.warning(f"Error processing frame {i}: {str(e)}")
            # Use zero shift as fallback
            if method == 'rigid':
                shifts[i] = [0, 0]
    
    total_time = time.time() - start_time
    if logger:
        logger.info(f"Motion estimation completed in {total_time:.2f} seconds")
    
    return shifts


def save_motion_corrected_data(corrected_data, shifts, output_path, original_path=None, params=None, logger=None):
    """
    Save motion-corrected data and shifts to file.
    
    Parameters
    ----------
    corrected_data : numpy.ndarray
        Motion-corrected data
    shifts : numpy.ndarray
        Calculated motion shifts
    output_path : str
        Path to save the data
    original_path : str, optional
        Path to the original data
    params : dict, optional
        Motion correction parameters
    logger : logging.Logger, optional
        Logger for status updates
    """
    try:
        # Save as HDF5 for maximum compatibility
        with h5py.File(output_path, 'w') as f:
            # Save corrected data
            f.create_dataset('corrected_data', data=corrected_data, compression='gzip')
            
            # Save shifts
            f.create_dataset('shifts', data=shifts)
            
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['correction_time'] = time.ctime()
            
            if original_path:
                meta_group.attrs['original_file'] = os.path.basename(original_path)
            
            if params:
                params_group = meta_group.create_group('params')
                for key, value in params.items():
                    if isinstance(value, (int, float, str)):
                        params_group.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        params_group.create_dataset(key, data=np.array(value))
        
        if logger:
            logger.info(f"Saved motion-corrected data to {output_path}")
        
        # Additionally save as TIFF if requested
        if output_path.endswith('.h5') or output_path.endswith('.hdf5'):
            tif_path = output_path.rsplit('.', 1)[0] + '.tif'
            tifffile.imwrite(tif_path, corrected_data)
            
            if logger:
                logger.info(f"Saved motion-corrected TIFF to {tif_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error saving motion-corrected data: {str(e)}")


def plot_motion_shifts(shifts, method='rigid', output_path=None, logger=None):
    """
    Plot motion shifts for visualization.
    
    Parameters
    ----------
    shifts : numpy.ndarray
        Calculated motion shifts
    method : str, optional
        Motion correction method ('rigid' or 'nonrigid')
    output_path : str, optional
        Path to save the plot
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    try:
        import matplotlib.pyplot as plt
        
        if method == 'rigid':
            # Rigid shifts have shape (n_frames, 2)
            n_frames = shifts.shape[0]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot X and Y shifts
            ax1.plot(shifts[:, 0], 'r-', label='X shifts')
            ax1.plot(shifts[:, 1], 'b-', label='Y shifts')
            ax1.set_title('Rigid Motion Shifts')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Shift (pixels)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot shift magnitude
            shift_magnitude = np.sqrt(shifts[:, 0]**2 + shifts[:, 1]**2)
            ax2.plot(shift_magnitude, 'g-')
            ax2.set_title('Motion Magnitude')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Magnitude (pixels)')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            avg_magnitude = np.mean(shift_magnitude)
            max_magnitude = np.max(shift_magnitude)
            
            fig.suptitle(f'Motion Analysis (Avg: {avg_magnitude:.2f} px, Max: {max_magnitude:.2f} px)')
            
        else:
            # Non-rigid shifts have shape (n_frames, n_patches_y, n_patches_x, 2)
            n_frames = shifts.shape[0]
            
            # Calculate average shift across patches for each frame
            avg_shifts = np.mean(shifts, axis=(1, 2))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot average X and Y shifts
            ax1.plot(avg_shifts[:, 0], 'r-', label='Avg X shifts')
            ax1.plot(avg_shifts[:, 1], 'b-', label='Avg Y shifts')
            ax1.set_title('Non-Rigid Average Motion Shifts')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Shift (pixels)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot average shift magnitude
            shift_magnitude = np.sqrt(avg_shifts[:, 0]**2 + avg_shifts[:, 1]**2)
            ax2.plot(shift_magnitude, 'g-')
            ax2.set_title('Average Motion Magnitude')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Magnitude (pixels)')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            avg_magnitude = np.mean(shift_magnitude)
            max_magnitude = np.max(shift_magnitude)
            
            fig.suptitle(f'Motion Analysis (Avg: {avg_magnitude:.2f} px, Max: {max_magnitude:.2f} px)')
        
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=150)
            
            if logger:
                logger.info(f"Saved motion shifts plot to {output_path}")
        
        return fig
        
    except Exception as e:
        if logger:
            logger.error(f"Error plotting motion shifts: {str(e)}")
        return None


def correct_motion_in_file(input_path, output_path, params, logger=None):
    """
    Apply motion correction to a TIFF file and save the result.
    
    Parameters
    ----------
    input_path : str
        Path to input TIFF file
    output_path : str
        Path to save corrected output
    params : dict
        Motion correction parameters
    logger : logging.Logger, optional
        Logger for status updates
        
    Returns
    -------
    tuple
        (success, shifts) - success flag and calculated shifts
    """
    if not HAS_NORMCORRE:
        if logger:
            logger.error("NoRMCorre not installed. Install with: pip install normcorre")
        return False, None
    
    try:
        if logger:
            logger.info(f"Loading {input_path} for motion correction")
        
        # Load data
        with tifffile.TiffFile(input_path) as tif:
            image_data = tif.asarray()
            
            # Ensure data is in (frames, height, width) format
            if len(image_data.shape) > 3:
                if logger:
                    logger.warning(f"Input data has {len(image_data.shape)} dimensions, reshaping")
                
                # Handle different formats
                if len(image_data.shape) == 4:  # Likely TZYX
                    # Check which dimension is smallest
                    if image_data.shape[0] < image_data.shape[1]:  # TZYX
                        image_data = image_data[:, 0]  # Take first channel
                    else:  # ZCYX or similar
                        image_data = image_data[0]  # Take first time/channel
                elif len(image_data.shape) == 5:  # Likely TZCYX
                    if image_data.shape[0] < image_data.shape[1]:  # TZCYX
                        image_data = image_data[:, 0, 0]  # Take first channel and first z-slice
                    else:  # Some other format
                        image_data = image_data[0, 0]  # Take first time and first channel
            
            # Ensure data is in expected format after reshaping
            if len(image_data.shape) != 3:
                if len(image_data.shape) == 2:  # Single frame
                    # Add dimension for frames
                    image_data = np.expand_dims(image_data, axis=0)
                else:
                    raise ValueError(f"Cannot reshape data with shape {image_data.shape} to (frames, height, width)")
            
            if logger:
                logger.info(f"Loaded data with shape {image_data.shape}")
        
        # Run motion correction
        corrected_data, shifts = apply_normcorre_correction(image_data, params, logger)
        
        # Save results
        save_motion_corrected_data(corrected_data, shifts, output_path, input_path, params, logger)
        
        # Generate motion plot
        plot_path = output_path.rsplit('.', 1)[0] + '_motion.png'
        plot_motion_shifts(shifts, params.get('method', 'rigid'), plot_path, logger)
        
        return True, shifts
        
    except Exception as e:
        if logger:
            logger.error(f"Error in motion correction: {str(e)}")
        return False, None