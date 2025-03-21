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
    logger,
    output_dir=None,
    metadata=None  # Add metadata parameter to access condition
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
    output_dir : str, optional
        Directory to save intermediate traces
    metadata : dict, optional
        Metadata dictionary with condition information
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing extracted metrics for each ROI
    """
    start_time = time.time()
    
    # Get condition from metadata if available
    condition = metadata.get("condition", "unknown") if metadata else "unknown"
    logger.info(f"Processing data for condition: {condition}")
    
    # Select condition-specific parameters if available
    if condition in config.get("condition_specific", {}):
        logger.info(f"Using condition-specific parameters for {condition}")
        condition_config = config["condition_specific"][condition]
        
        # Extract parameters from condition-specific config
        baseline_frames = condition_config.get("baseline_frames", config.get("baseline_frames", [0, 100]))
        analysis_frames = condition_config.get("analysis_frames", config.get("analysis_frames", [100, 580]))
        active_threshold = condition_config.get("active_threshold", config.get("active_threshold", 0.02))
        active_metric = condition_config.get("active_metric", "peak_amplitude")
        
        logger.info(f"Condition {condition}: Analysis frames {analysis_frames}, active metric: {active_metric}")
    else:
        # Use default parameters if condition-specific not found
        logger.info(f"No specific parameters for condition {condition}, using defaults")
        baseline_frames = config.get("baseline_frames", [0, 100])
        analysis_frames = config.get("analysis_frames", [100, 580])  
        active_threshold = config.get("active_threshold", 0.02)
        active_metric = "peak_amplitude"  # Default metric
    
    logger.info(f"Starting fluorescence analysis for condition: {condition}")
    
    n_rois, n_frames = fluorescence_data.shape

    # Get save_intermediate_traces flag from config
    save_intermediate = config.get("save_intermediate_traces", False)

    # Check if we have a valid output directory if saving is enabled
    if save_intermediate and output_dir is None:
        logger.warning("save_intermediate_traces is enabled but no output_dir provided. Disabling intermediate saves.")
        save_intermediate = False
    
    # Create directory for intermediate traces if needed
    if save_intermediate:
        traces_dir = os.path.join(output_dir, "intermediate_traces")
        os.makedirs(traces_dir, exist_ok=True)
        logger.info(f"Will save intermediate traces to {traces_dir}")
        
        # Save raw traces before any processing
        raw_traces_path = os.path.join(traces_dir, "0_raw_traces.csv")
        pd.DataFrame(fluorescence_data).to_csv(raw_traces_path)
        logger.info(f"Saved raw traces to {raw_traces_path}")
    
    # Verify frame ranges are valid
    baseline_frames = [max(0, baseline_frames[0]), min(n_frames-1, baseline_frames[1])]
    analysis_frames = [max(0, analysis_frames[0]), min(n_frames-1, analysis_frames[1])]
    
    logger.info(f"Using frames {baseline_frames[0]}-{baseline_frames[1]} for baseline calculation")
    
    # Get baseline calculation method and parameters
    baseline_method = config.get("baseline_method", "percentile")  # Default to percentile method
    baseline_percentile = config.get("baseline_percentile", 8)  # Default to 8th percentile
    baseline_n_frames = config.get("baseline_n_frames", 10)  # Default to first 10 frames (for mean method)
    
    if baseline_method == "percentile":
        logger.info(f"Using {baseline_percentile}th percentile for baseline calculation")
    elif baseline_method == "mean":
        logger.info(f"Using average of first {baseline_n_frames} frames for baseline calculation")
    logger.info(f"Using frames {analysis_frames[0]}-{analysis_frames[1]} for analysis")
    
    # Create a DataFrame to store metrics
    metrics = []
    
    # Calculate distance to lamina for each ROI
    lamina_distances = measure_distance_to_lamina(roi_masks, logger)
    
    # Arrays to store intermediate traces if saving is enabled
    if save_intermediate:
        photobleach_corrected = np.zeros_like(fluorescence_data)
        df_f_traces = np.zeros_like(fluorescence_data)
    
    # Process each ROI
    for i in range(n_rois):
        # Get trace for this ROI
        trace = fluorescence_data[i]
        
        # Apply photobleaching correction based on first 100 frames (excluding peaks)
        corrected_trace = correct_photobleaching(trace, baseline_frames, logger)
        
        # Save photobleaching-corrected trace if enabled
        if save_intermediate:
            photobleach_corrected[i] = corrected_trace
        
        # Calculate baseline based on selected method
        if baseline_method == "percentile":
            # Use percentile method
            baseline = calculate_baseline_excluding_peaks(
                corrected_trace, 
                baseline_frames, 
                logger, 
                percentile=baseline_percentile
            )
            logger.info(f"ROI {i+1} baseline ({baseline_percentile}th percentile): {baseline:.4f}")
        elif baseline_method == "mean":
            # Use mean of first N frames
            f0_frames = min(baseline_n_frames, n_frames)  # Ensure we don't exceed available frames
            baseline = np.mean(corrected_trace[:f0_frames])
            logger.info(f"ROI {i+1} baseline (average of first {f0_frames} frames): {baseline:.4f}")
        elif baseline_method == "min":
            # Use minimum of baseline window
            baseline_window = corrected_trace[baseline_frames[0]:baseline_frames[1]+1]
            baseline = np.min(baseline_window)
            logger.info(f"ROI {i+1} baseline (minimum value): {baseline:.4f}")
        else:
            # Default to percentile method if unknown
            logger.warning(f"Unknown baseline method '{baseline_method}', falling back to percentile")
            baseline = calculate_baseline_excluding_peaks(
                corrected_trace, 
                baseline_frames, 
                logger, 
                percentile=baseline_percentile
            )
            logger.info(f"ROI {i+1} baseline ({baseline_percentile}th percentile): {baseline:.4f}")
        
        # Calculate dF/F
        df_f = (corrected_trace - baseline) / baseline if baseline != 0 else corrected_trace
        
        # Save dF/F trace if enabled
        if save_intermediate:
            df_f_traces[i] = df_f
        
        # Extract analysis window
        analysis_window = df_f[analysis_frames[0]:analysis_frames[1]+1]
        analysis_time = np.arange(len(analysis_window))
        
        # Calculate basic statistics
        mean_df_f = np.mean(analysis_window)
        max_df_f = np.max(analysis_window)
        min_df_f = np.min(analysis_window)
        std_df_f = np.std(analysis_window)
        
        # Find evoked peak parameters - always calculate them
        peak_params = extract_peak_parameters(
            analysis_window, 
            config.get("peak_detection", {}),
            logger
        )
        
        # Find spontaneous activity - always calculate them
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
            
        # Determine whether the ROI is active based on condition-specific metric
        # For 0um condition: use spontaneous frequency
        # For 10um and 25um conditions: use peak amplitude
        if active_metric == "spont_peak_frequency":
            is_active = spont_params['peak_frequency'] > active_threshold
            logger.info(f"ROI {i+1} activity determined by spontaneous peak frequency: {spont_params['peak_frequency']:.4f} (threshold: {active_threshold})")
        elif active_metric == "peak_amplitude":
            is_active = peak_params['amplitude'] > active_threshold
            logger.info(f"ROI {i+1} activity determined by peak amplitude: {peak_params['amplitude']:.4f} (threshold: {active_threshold})")
        else:
            # Default to peak amplitude if unknown metric
            is_active = peak_params['amplitude'] > active_threshold
            logger.warning(f"Unknown active_metric '{active_metric}', using peak amplitude")
        
        # Compile metrics for this ROI
        roi_metrics = {
            'roi_id': i + 1,
            'distance_to_lamina': lamina_distances[i],
            'baseline_fluorescence': baseline,
            'mean_df_f': mean_df_f,
            'max_df_f': max_df_f,
            'min_df_f': min_df_f,
            'std_df_f': std_df_f,
            'is_active': is_active,
            'condition': condition  # Include condition in metrics
        }
        
        # Add peak parameters
        roi_metrics.update({f'peak_{k}': v for k, v in peak_params.items()})
        
        # Add spontaneous activity parameters
        roi_metrics.update({f'spont_{k}': v for k, v in spont_params.items()})
        
        metrics.append(roi_metrics)
    
    # Save intermediate traces after all ROIs are processed
    if save_intermediate:
        # Save photobleaching-corrected traces
        pb_traces_path = os.path.join(traces_dir, "1_photobleach_corrected.csv")
        pd.DataFrame(photobleach_corrected).to_csv(pb_traces_path)
        logger.info(f"Saved photobleaching-corrected traces to {pb_traces_path}")
        
        # Save dF/F traces
        df_f_traces_path = os.path.join(traces_dir, "2_df_f_traces.csv")
        pd.DataFrame(df_f_traces).to_csv(df_f_traces_path)
        logger.info(f"Saved dF/F traces to {df_f_traces_path}")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    logger.info(f"Fluorescence analysis completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Extracted metrics for {n_rois} ROIs")
    
    return metrics_df

def correct_photobleaching(trace, baseline_frames, logger):
    """
    Estimate the slope of photobleaching from the first 200 frames, excluding peaks.
    Apply a correction to flatten this trend throughout the entire trace,
    regardless of whether the slope is positive or negative.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Original fluorescence trace
    baseline_frames : list
        Range of frames to use for baseline correction [start, end]
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    numpy.ndarray
        Photobleaching-corrected trace with flattened baseline
    """
    try:
        # Extend baseline window to 200 frames or use all available frames if less than 200
        extended_frames = [0, min(200, len(trace)-1)]
        baseline_window = trace[extended_frames[0]:extended_frames[1]+1]
        baseline_x = np.arange(len(baseline_window))
        
        # Find peaks in the baseline window to exclude them
        peaks, _ = find_peaks(baseline_window, prominence=0.05)
        
        # Create mask to exclude peaks and their surrounding frames (±2 frames)
        mask = np.ones(len(baseline_window), dtype=bool)
        for peak in peaks:
            start = max(0, peak - 2)
            end = min(len(baseline_window), peak + 3)  # +3 because slicing is exclusive of end
            mask[start:end] = False
        
        # If all frames would be excluded, keep at least half of them
        if not np.any(mask) and len(baseline_window) > 0:
            logger.warning("All baseline frames would be excluded. Keeping 50% of frames.")
            mask = np.ones(len(baseline_window), dtype=bool)
            for peak in peaks:
                mask[peak] = False  # Just exclude the exact peak
        
        # Fit a line to the non-peak frames to estimate photobleaching slope
        if np.sum(mask) > 1:  # Need at least 2 points for linear regression
            x_fit = baseline_x[mask]
            y_fit = baseline_window[mask]
            
            # Use polyfit for linear regression: y = mx + b
            m, b = np.polyfit(x_fit, y_fit, 1)
            
            # If slope is essentially flat, don't correct
            if abs(m) < 1e-5:
                logger.info("No significant trend detected. Slope is nearly flat.")
                return trace.copy()
            
            # Calculate the trend using the estimated slope and intercept
            x_all = np.arange(len(trace))
            trend = m * x_all + b
            
            # Calculate the mean of the baseline points used for fitting
            baseline_mean = np.mean(baseline_window[mask])
            
            # Create a flat baseline at the mean level
            flat_baseline = np.ones(len(trace)) * baseline_mean
            
            # Replace the trended baseline with a flat baseline
            # Preserve the fluctuations around the trend line
            corrected_trace = trace - trend + flat_baseline
            
            if m < 0:
                logger.info(f"Negative slope detected ({m:.6f}). Applied correction to flatten baseline.")
            else:
                logger.info(f"Positive slope detected ({m:.6f}). Applied correction to flatten baseline.")
                
            return corrected_trace
        else:
            logger.warning("Not enough non-peak points to estimate trend. Using original trace.")
            return trace.copy()
    
    except Exception as e:
        logger.error(f"Error in trend correction: {str(e)}. Using original trace.")
        return trace.copy()

def calculate_baseline_excluding_peaks(trace, baseline_frames, logger, percentile=8):
    """
    Calculate baseline using the 8th percentile of the first 100 frames, excluding peaks.
    Using a lower percentile helps avoid overestimation of baseline when there are 
    spontaneous events during the baseline period.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Fluorescence trace (already photobleaching-corrected)
    baseline_frames : list
        Range of frames to use for baseline [start, end]
    logger : logging.Logger
        Logger object
    percentile : int, optional
        Percentile to use as baseline (default: 8)
        
    Returns
    -------
    float
        Baseline fluorescence value
    """
    try:
        # Extract baseline window
        baseline_window = trace[baseline_frames[0]:baseline_frames[1]+1]
        
        # Find peaks in the baseline window to exclude them
        peaks, _ = find_peaks(baseline_window, prominence=0.05)
        
        # Create mask to exclude peaks and their surrounding frames (±2 frames)
        mask = np.ones(len(baseline_window), dtype=bool)
        for peak in peaks:
            start = max(0, peak - 2)
            end = min(len(baseline_window), peak + 3)  # +3 because slicing is exclusive of end
            mask[start:end] = False
        
        # If all frames would be excluded, keep at least half of them
        if not np.any(mask) and len(baseline_window) > 0:
            logger.warning("All baseline frames would be excluded. Using standard percentile calculation.")
            return np.percentile(baseline_window, percentile)
        
        # Calculate 8th percentile of non-peak frames
        if np.sum(mask) > 0:
            filtered_baseline = baseline_window[mask]
            baseline = np.percentile(filtered_baseline, percentile)
            logger.info(f"Calculated baseline using {percentile}th percentile after excluding {len(peaks)} peaks: {baseline:.4f}")
            return baseline
        else:
            # Fallback if something went wrong with the mask
            baseline = np.percentile(baseline_window, percentile)
            logger.warning(f"Using {percentile}th percentile of all baseline frames: {baseline:.4f}")
            return baseline
    
    except Exception as e:
        logger.error(f"Error in baseline calculation: {str(e)}. Using {percentile}th percentile of all frames.")
        return np.percentile(trace[baseline_frames[0]:baseline_frames[1]+1], percentile)

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
    # Get peak detection parameters with more sensitivity options
    prominence = peak_config.get("prominence", 0.03)  # Reduced default from 0.1 to 0.03
    width = peak_config.get("width", 2)               # Reduced default from 3 to 2
    # Additional parameters for find_peaks
    distance = peak_config.get("distance", 10)        # Minimum distance between peaks
    height = peak_config.get("height", 0.01)          # Minimum peak height
    rel_height = peak_config.get("rel_height", 0.5)   # Relative height for width calculation
    
    # Apply light smoothing to help with peak detection
    try:
        # Only smooth if we have enough points
        if len(trace) > 5:
            # Apply Savitzky-Golay filter for noise reduction (preserves peak shapes)
            smooth_trace = savgol_filter(trace, min(5, len(trace)), 2)
        else:
            smooth_trace = trace
            
        # Find peaks with enhanced parameters
        peaks, properties = find_peaks(
            smooth_trace,
            prominence=prominence,
            width=width,
            distance=distance,
            height=height,
            rel_height=rel_height
        )
        
        # Log number of peaks found
        if len(peaks) > 0:
            logger.info(f"Found {len(peaks)} peaks with max prominence: {max(properties['prominences']):.4f}")
        else:
            logger.info("No peaks found in trace")
            
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
    peak_amplitude = smooth_trace[peak_idx]  # Using the smoothed trace for more accurate amplitude
    
    # Get time of peak (frame number)
    time_of_peak = peak_idx
    
    # Calculate rise slope
    try:
        left_base = int(properties['left_bases'][max_peak_idx])
        
        # Ensure valid range
        if left_base == peak_idx:
            left_base = max(0, peak_idx - 1)
        
        # Calculate slope for each point during rise phase
        rise_phase = smooth_trace[left_base:peak_idx+1]
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
            
            # Use integrate.simps for more accurate area calculation
            x_range = np.arange(left_base, right_base+1)
            peak_segment = smooth_trace[left_base:right_base+1]
            
            # Calculate area using Simpson's rule
            auc = integrate.simps(peak_segment, x_range)
        except Exception as e:
            logger.warning(f"Error calculating AUC: {str(e)}")
            auc = np.sum(smooth_trace[left_base:right_base+1])
    except Exception as e:
        logger.warning(f"Error calculating peak parameters: {str(e)}")
        max_rise_slope = 0
        time_of_max_rise = 0
        rise_time = 0
        auc = 0
    
    # Calculate additional metrics
    try:
        # Find all peaks amplitude stats
        all_peak_amplitudes = smooth_trace[peaks]
        mean_peak_amplitude = np.mean(all_peak_amplitudes)
        max_amplitude = np.max(all_peak_amplitudes)
        
        # Calculate peak frequency (per 100 frames)
        peak_frequency = len(peaks) / (len(trace) / 100)
    except Exception:
        mean_peak_amplitude = peak_amplitude
        max_amplitude = peak_amplitude
        peak_frequency = 1 if peak_amplitude > 0 else 0
    
    return {
        'amplitude': peak_amplitude,
        'max_amplitude': max_amplitude,           # Added max amplitude across all peaks
        'mean_amplitude': mean_peak_amplitude,    # Added mean amplitude across all peaks
        'peak_count': len(peaks),                 # Added count of peaks
        'peak_frequency': peak_frequency,         # Added peak frequency
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