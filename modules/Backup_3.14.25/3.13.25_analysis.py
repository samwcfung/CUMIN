#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluorescence Analysis Module
--------------------------
Extract metrics from fluorescence traces, performs QC checks.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import integrate
from pathlib import Path
import tifffile

def analyze_fluorescence(
    fluorescence_data, 
    roi_masks, 
    original_image_path,
    config, 
    logger
):
    """
    Extract metrics from fluorescence traces.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks
    original_image_path : str
        Path to the original .tif image (for measuring distance to lamina)
    config : dict
        Configuration parameters
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing extracted metrics for each ROI
    """
    start_time = time.time()
    logger.info("Starting fluorescence analysis")
    
    n_rois, n_frames = fluorescence_data.shape
    
    # Extract baseline and analysis frame ranges
    baseline_frames = config.get("baseline_frames", [0, 200])
    analysis_frames = config.get("analysis_frames", [233, 580])
    
    # Verify frame ranges are valid
    baseline_frames = [max(0, baseline_frames[0]), min(n_frames-1, baseline_frames[1])]
    analysis_frames = [max(0, analysis_frames[0]), min(n_frames-1, analysis_frames[1])]
    
    logger.info(f"Using frames {baseline_frames[0]}-{baseline_frames[1]} for baseline")
    logger.info(f"Using frames {analysis_frames[0]}-{analysis_frames[1]} for analysis")
    
    # Create a DataFrame to store metrics
    metrics = []
    
    # Calculate distance to lamina for each ROI
    lamina_distances = measure_distance_to_lamina(roi_masks, logger)
    
    # Process each ROI
    for i in range(n_rois):
        # Get trace for this ROI
        trace = fluorescence_data[i]
        
        # Calculate baseline
        baseline = np.mean(trace[baseline_frames[0]:baseline_frames[1]+1])
        
        # Calculate dF/F
        df_f = (trace - baseline) / baseline if baseline != 0 else trace
        
        # Extract analysis window
        analysis_window = df_f[analysis_frames[0]:analysis_frames[1]+1]
        analysis_time = np.arange(len(analysis_window))
        
        # Calculate basic statistics
        mean_df_f = np.mean(analysis_window)
        max_df_f = np.max(analysis_window)
        min_df_f = np.min(analysis_window)
        std_df_f = np.std(analysis_window)
        
        # Find evoked peak parameters
        peak_params = extract_peak_parameters(
            analysis_window, 
            config.get("peak_detection", {}),
            logger
        )
        
        # Find spontaneous activity
        spont_params = extract_spontaneous_activity(
            df_f[baseline_frames[0]:baseline_frames[1]+1],
            config.get("spontaneous_activity", {}),
            logger
        )
        
        # Make sure both possible keys are handled
        if 'amplitude' not in peak_params:
            peak_params['amplitude'] = 0.0
        if 'peak_amplitude' not in peak_params:
            peak_params['peak_amplitude'] = peak_params['amplitude']
            
        # Determine whether the ROI is active - use try/except to be safe
        try:
            is_active = peak_params['amplitude'] > config.get("active_threshold", 0.1)
        except Exception as e:
            logger.warning(f"Error determining ROI activity: {str(e)}. Setting to inactive.")
            is_active = False
        
        # Compile metrics for this ROI
        roi_metrics = {
            'roi_id': i + 1,
            'distance_to_lamina': lamina_distances[i],
            'baseline_fluorescence': baseline,
            'mean_df_f': mean_df_f,
            'max_df_f': max_df_f,
            'min_df_f': min_df_f,
            'std_df_f': std_df_f,
            'is_active': is_active
        }
        
        # Add peak parameters
        roi_metrics.update({f'peak_{k}': v for k, v in peak_params.items()})
        
        # Add spontaneous activity parameters
        roi_metrics.update({f'spont_{k}': v for k, v in spont_params.items()})
        
        metrics.append(roi_metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    logger.info(f"Fluorescence analysis completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Extracted metrics for {n_rois} ROIs")
    
    return metrics_df

def measure_distance_to_lamina(roi_masks, logger):
    """
    Measure the distance from each ROI center to the top of the image.
    
    Parameters
    ----------
    roi_masks : list
        List of ROI masks
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    list
        List of distances (in pixels) from ROI center to top of image
    """
    distances = []
    
    for mask in roi_masks:
        # Find ROI center
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            
            # Distance to top (lamina border)
            distance = center_y
        else:
            # Default if mask is empty
            distance = 0
            logger.warning("Empty mask found when calculating distance to lamina")
        
        distances.append(distance)
    
    return distances

def extract_peak_parameters(
    trace, 
    peak_config, 
    logger
):
    """
    Extract peak parameters from a fluorescence trace.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Fluorescence trace (dF/F)
    peak_config : dict
        Peak detection configuration
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    dict
        Dictionary of peak parameters
    """
    # Get peak detection parameters
    prominence = peak_config.get("prominence", 0.1)
    width = peak_config.get("width", 3)
    
    # Find peaks
    try:
        peaks, properties = find_peaks(
            trace, 
            prominence=prominence,
            width=width
        )
    except Exception as e:
        logger.warning(f"Error in peak detection: {str(e)}")
        # Return default values if peak detection fails
        return {
            'amplitude': 0.0,
            'time_of_peak': 0,
            'max_rise_slope': 0.0,
            'time_of_max_rise': 0.0,
            'rise_time': 0.0,
            'area_under_curve': 0.0
        }
    
    # If no peaks found, return zeros
    if len(peaks) == 0:
        return {
            'amplitude': 0.0,
            'time_of_peak': 0,
            'max_rise_slope': 0.0,
            'time_of_max_rise': 0.0,
            'rise_time': 0.0,
            'area_under_curve': 0.0
        }
    
    # Find the most prominent peak
    max_peak_idx = np.argmax(properties['prominences'])
    peak_idx = peaks[max_peak_idx]
    
    # Get peak amplitude
    peak_amplitude = trace[peak_idx]
    
    # Get time of peak (frame number)
    time_of_peak = peak_idx
    
    # Calculate rise slope
    try:
        left_base = int(properties['left_bases'][max_peak_idx])
        
        # Ensure valid range
        if left_base == peak_idx:
            left_base = max(0, peak_idx - 1)
        
        # Calculate slope for each point during rise phase
        rise_phase = trace[left_base:peak_idx+1]
        rise_times = np.arange(len(rise_phase))
        
        # Apply Savitzky-Golay filter for smoother derivative calculation
        if len(rise_phase) > 5:  # Only if we have enough points
            rise_phase_smooth = savgol_filter(rise_phase, min(5, len(rise_phase)), 2)
        else:
            rise_phase_smooth = rise_phase
        
        # Calculate numerical derivative
        slopes = np.diff(rise_phase_smooth) / np.diff(rise_times) if len(rise_times) > 1 else [0]
        
        # Find maximum rise slope
        max_rise_slope = np.max(slopes) if len(slopes) > 0 else 0
        time_of_max_rise = left_base + np.argmax(slopes) if len(slopes) > 0 else 0
        
        # Calculate rise time (time from 10% to 90% of peak amplitude)
        try:
            rise_start = next((i for i, v in enumerate(rise_phase) if v >= 0.1 * peak_amplitude), 0)
            rise_end = next((i for i, v in enumerate(rise_phase) if v >= 0.9 * peak_amplitude), len(rise_phase)-1)
            rise_time = rise_end - rise_start
        except Exception:
            rise_time = 0
        
        # Calculate area under curve
        try:
            right_base = int(properties['right_bases'][max_peak_idx])
            auc = np.sum(trace[left_base:right_base+1])
        except Exception:
            auc = 0
    except Exception as e:
        logger.warning(f"Error calculating peak parameters: {str(e)}")
        max_rise_slope = 0
        time_of_max_rise = 0
        rise_time = 0
        auc = 0
    
    return {
        'amplitude': peak_amplitude,
        'time_of_peak': time_of_peak,
        'max_rise_slope': max_rise_slope,
        'time_of_max_rise': time_of_max_rise,
        'rise_time': rise_time,
        'area_under_curve': auc
    }

def extract_spontaneous_activity(
    trace, 
    spont_config, 
    logger
):
    """
    Extract spontaneous activity parameters from baseline fluorescence.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Fluorescence trace during baseline period (dF/F)
    spont_config : dict
        Spontaneous activity detection configuration
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    dict
        Dictionary of spontaneous activity parameters
    """
    # Get spontaneous activity detection parameters
    prominence = spont_config.get("prominence", 0.05)
    width = spont_config.get("width", 2)
    
    # Find spontaneous peaks
    try:
        peaks, properties = find_peaks(
            trace, 
            prominence=prominence,
            width=width
        )
    except Exception as e:
        logger.warning(f"Error in spontaneous peak detection: {str(e)}")
        peaks = []
        properties = {'prominences': []}
    
    # Calculate peak frequency (peaks per 100 frames)
    peak_frequency = len(peaks) / (len(trace) / 100) if len(trace) > 0 else 0
    
    # Calculate average peak amplitude
    if len(peaks) > 0:
        avg_peak_amplitude = np.mean(trace[peaks])
        peak_std = np.std(trace[peaks])
        max_peak = np.max(trace[peaks])
    else:
        avg_peak_amplitude = 0.0
        peak_std = 0.0
        max_peak = 0.0
    
    return {
        'peak_frequency': peak_frequency,
        'avg_peak_amplitude': avg_peak_amplitude,
        'peak_std': peak_std,
        'max_peak': max_peak
    }

def perform_qc_checks(fluorescence_data, metrics_df, qc_thresholds, logger):
    """
    Perform quality control checks on ROI fluorescence data.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    qc_thresholds : dict
        QC threshold parameters
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    list
        List of flagged ROI IDs with issues
    """
    flagged_rois = []
    n_rois = fluorescence_data.shape[0]
    
    logger.info("Performing QC checks on ROI data")
    
    # Check for abnormally low fluorescence variance
    min_variance = qc_thresholds.get("min_variance", 0.01)
    
    for i in range(n_rois):
        # Calculate variance of trace
        variance = np.var(fluorescence_data[i])
        
        if variance < min_variance:
            flagged_rois.append({
                'roi_id': i + 1,
                'issue': 'low_variance',
                'value': variance,
                'threshold': min_variance
            })
    
    # Check for potential motion artifacts
    # 1. Sudden jumps in fluorescence intensity
    max_jump = qc_thresholds.get("max_jump", 0.5)
    
    for i in range(n_rois):
        # Calculate frame-to-frame differences
        diffs = np.abs(np.diff(fluorescence_data[i]))
        max_diff = np.max(diffs) if len(diffs) > 0 else 0
        
        if max_diff > max_jump:
            flagged_rois.append({
                'roi_id': i + 1,
                'issue': 'intensity_jump',
                'value': max_diff,
                'threshold': max_jump,
                'frame': np.argmax(diffs) + 1  # Add 1 because diff reduces length by 1
            })
    
    # 2. Baseline drift
    max_drift = qc_thresholds.get("max_drift", 0.3)
    
    for i in range(n_rois):
        # Calculate drift as difference between mean of first and last 10% of frames
        n_frames = fluorescence_data.shape[1]
        n_edge = max(int(n_frames * 0.1), 1)
        
        first_mean = np.mean(fluorescence_data[i, :n_edge])
        last_mean = np.mean(fluorescence_data[i, -n_edge:])
        
        drift = np.abs(last_mean - first_mean) / first_mean if first_mean != 0 else 0
        
        if drift > max_drift:
            flagged_rois.append({
                'roi_id': i + 1,
                'issue': 'baseline_drift',
                'value': drift,
                'threshold': max_drift
            })
    
    # Log flagged ROIs
    logger.info(f"Flagged {len(flagged_rois)} ROIs with potential issues")
    
    for flag in flagged_rois:
        logger.info(f"ROI {flag['roi_id']}: {flag['issue']} ({flag['value']:.4f} > {flag['threshold']:.4f})")
    
    return flagged_rois