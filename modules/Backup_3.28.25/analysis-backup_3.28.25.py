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
    metadata=None
):
    """
    Extract metrics from fluorescence traces with condition-specific processing.
    """
    start_time = time.time()
    
    # Get condition from metadata if available
    condition = metadata.get("condition", "unknown") if metadata else "unknown"
    logger.info(f"Processing data for condition: {condition}")
    
    # Check if data has already been preprocessed
    use_preprocessed_data = config.get("use_preprocessed_data", True)
    
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
        df_f_traces = np.zeros_like(fluorescence_data)
    
    # Create array to store dF/F traces for all ROIs
    df_f_all_traces = np.zeros_like(fluorescence_data)
    
    # Process each ROI
    for i in range(n_rois):
        # Get trace for this ROI
        trace = fluorescence_data[i]
        
        # If using pre-processed data, we can skip photobleaching correction
        # The data should already be photobleaching-corrected and background-subtracted
        if use_preprocessed_data:
            corrected_trace = trace  # Use as is, assuming it's already corrected
        else:
            # Apply photobleaching correction based on first 100 frames (excluding peaks)
            corrected_trace = correct_photobleaching(trace, baseline_frames, logger, condition=condition, config=config)
            
            # Save photobleaching-corrected trace if enabled
            if save_intermediate:
                photobleach_corrected = np.zeros_like(fluorescence_data)
                photobleach_corrected[i] = corrected_trace
        
        # Calculate baseline based on selected method
        if baseline_method == "percentile":
            # Use percentile method - explicitly ensure 8th percentile
            logger.info(f"Using {baseline_percentile}th percentile for baseline calculation")
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
        
        # Store dF/F trace for this ROI
        df_f_all_traces[i] = df_f
        
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
        
        # ===== CONDITION-SPECIFIC ANALYSIS METHODS =====
        
        # Check condition for appropriate analysis approach
        if condition == "0um":
            # Spontaneous activity - use existing peak detection method
            logger.info(f"ROI {i+1}: Using spontaneous peak detection for 0um condition")
            
            # Use analysis_frames instead of baseline_frames for spontaneous activity detection
            spont_params = extract_spontaneous_activity(
                df_f[analysis_frames[0]:analysis_frames[1]+1],  # Use analysis frames instead of baseline frames
                config.get("spontaneous_activity", {}),
                config.get("peak_detection", {}),
                logger
            )
            
            # Also find peaks using standard method for metrics
            peak_params = extract_peak_parameters(
                analysis_window, 
                config.get("peak_detection", {}),
                logger
            )
            
            # Make sure both possible keys are handled for backward compatibility
            if 'amplitude' not in peak_params:
                peak_params['amplitude'] = 0.0
            if 'peak_amplitude' not in peak_params:
                peak_params['peak_amplitude'] = peak_params['amplitude']
            
            # Determine activity based on spontaneous frequency from the analysis window
            is_active = spont_params['peak_frequency'] > active_threshold
            activity_value = spont_params['peak_frequency'] 
            logger.info(f"ROI {i+1} activity determined by spontaneous peak frequency: {activity_value:.4f} (threshold: {active_threshold})")
            
        else:  # 10um and 25um conditions
            # Evoked activity - use maximum dF/F and analyze the surrounding curve
            logger.info(f"ROI {i+1}: Using evoked response detection for {condition} condition")
            
            # Still calculate spontaneous parameters for completeness (using baseline period)
            spont_params = extract_spontaneous_activity(
                df_f[baseline_frames[0]:baseline_frames[1]+1],
                config.get("spontaneous_activity", {}),
                config.get("peak_detection", {}),
                logger
            )
            
            # Extract evoked response parameters focusing on maximum dF/F
            # This replaces the traditional peak detection for evoked signals
            evoked_params = extract_evoked_response(
                analysis_window,
                config.get("evoked_detection", {}),
                logger
            )
            
            # Use these parameters for our metrics
            peak_params = {
                'amplitude': evoked_params['max_value'],
                'peak_amplitude': evoked_params['max_value'],
                'max_amplitude': evoked_params['max_value'],
                'mean_amplitude': evoked_params['mean_value'],
                'time_of_peak': evoked_params['max_frame'],
                'rise_time': evoked_params['rise_time'],
                'max_rise_slope': evoked_params['max_rise_slope'],
                'time_of_max_rise': evoked_params['time_of_max_rise'],
                'area_under_curve': evoked_params['area'],
                'peak_count': 1 if evoked_params['max_value'] > 0 else 0,
                'peak_frequency': 0  # Not relevant for evoked responses
            }
            
            # Determine activity based on maximum dF/F value
            is_active = evoked_params['max_value'] > active_threshold
            activity_value = evoked_params['max_value']
            logger.info(f"ROI {i+1} activity determined by maximum dF/F: {activity_value:.4f} (threshold: {active_threshold})")
        
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
    if save_intermediate and not use_preprocessed_data:
        # Save photobleaching-corrected traces (only if we did this step)
        pb_traces_path = os.path.join(traces_dir, "1_photobleach_corrected.csv")
        pd.DataFrame(photobleach_corrected).to_csv(pb_traces_path)
        logger.info(f"Saved photobleaching-corrected traces to {pb_traces_path}")
    
    # Always save dF/F traces
    if save_intermediate:
        # Save dF/F traces
        df_f_traces_path = os.path.join(traces_dir, "2_df_f_traces.csv")
        pd.DataFrame(df_f_all_traces).to_csv(df_f_traces_path)
        logger.info(f"Saved dF/F traces to {df_f_traces_path}")
    
    # ALWAYS save the dF/F traces to a dedicated file in the main output directory (not just in intermediate traces)
    df_f_output_path = os.path.join(output_dir, f"{Path(original_image_path).stem}_df_f_traces.csv")
    pd.DataFrame(df_f_all_traces).to_csv(df_f_output_path)
    logger.info(f"Saved dF/F traces (baseline normalized to 0) to {df_f_output_path}")
    
    # Also save as HDF5 for more efficient storage and faster loading
    df_f_h5_path = os.path.join(output_dir, f"{Path(original_image_path).stem}_df_f_traces.h5")
    try:
        import h5py
        with h5py.File(df_f_h5_path, 'w') as f:
            f.create_dataset('df_f_traces', data=df_f_all_traces, compression='gzip')
            # Store metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['condition'] = condition if condition else "unknown"
            meta_group.attrs['baseline_method'] = baseline_method
            meta_group.attrs['baseline_percentile'] = baseline_percentile
            meta_group.attrs['baseline_frames'] = baseline_frames
            meta_group.attrs['analysis_frames'] = analysis_frames
            meta_group.attrs['n_rois'] = n_rois
            meta_group.attrs['n_frames'] = n_frames
            meta_group.attrs['source_file'] = Path(original_image_path).name
            meta_group.attrs['use_preprocessed_data'] = use_preprocessed_data
        logger.info(f"Saved dF/F traces to HDF5 file {df_f_h5_path}")
    except Exception as e:
        logger.warning(f"Failed to save HDF5 file: {str(e)}")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    logger.info(f"Fluorescence analysis completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Extracted metrics for {n_rois} ROIs")
    
    return metrics_df, df_f_all_traces  # Return both

def extract_evoked_response(trace, config, logger):
    """
    Extract parameters related to an evoked response curve.
    This function focuses on the maximum dF/F point and analyzes 
    the surrounding curve characteristics.
    
    Parameters
    ----------
    trace : numpy.ndarray
        dF/F trace to analyze
    config : dict
        Configuration parameters for evoked response detection
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    dict
        Dictionary of evoked response parameters
    """
    try:
        # Find the maximum value and its location
        max_value = np.max(trace)
        max_frame = np.argmax(trace)
        
        # If no significant response, return default values
        if max_value <= 0.001:  # Small threshold to avoid division by zero issues
            logger.info("No significant evoked response detected")
            return {
                'max_value': 0.0,
                'max_frame': 0,
                'onset_frame': 0,
                'offset_frame': 0,
                'duration': 0,
                'area': 0.0,
                'rise_time': 0,
                'max_rise_slope': 0.0,
                'time_of_max_rise': 0,
                'mean_value': 0.0
            }
        
        # Get half-maximum value for rise/fall time calculations
        half_max = max_value / 2
        
        # Find the onset of the response (first point before max that goes below half max)
        onset_frame = 0
        for i in range(max_frame, 0, -1):
            if trace[i] < half_max:
                onset_frame = i + 1  # First point above half-max
                break
        
        # Find the offset of the response (first point after max that goes below half max)
        offset_frame = len(trace) - 1
        for i in range(max_frame, len(trace)):
            if trace[i] < half_max:
                offset_frame = i - 1  # Last point above half-max
                break
        
        # Calculate response duration (using half-max width)
        duration = offset_frame - onset_frame + 1
        
        # Calculate area under the curve for the full response
        # Use a threshold of 10% of max to define the full extent
        threshold = max_value * 0.1
        response_start = 0
        for i in range(max_frame, 0, -1):
            if trace[i] < threshold:
                response_start = i + 1
                break
        
        response_end = len(trace) - 1
        for i in range(max_frame, len(trace)):
            if trace[i] < threshold:
                response_end = i - 1
                break
        
        # Calculate the area - integrate over the thresholded response
        response_window = trace[response_start:response_end+1]
        area = np.sum(response_window)
        
        # Calculate mean value of the response during the main portion
        mean_value = np.mean(trace[onset_frame:offset_frame+1])
        
        # Calculate rise slope metrics
        rise_phase = trace[onset_frame:max_frame+1]
        rise_times = np.arange(len(rise_phase))
        
        # Rise time is from onset to peak
        rise_time = max_frame - onset_frame + 1
        
        # Calculate rise slopes
        if len(rise_phase) > 1:
            # Apply Savitzky-Golay filter for smoother derivative calculation if we have enough points
            if len(rise_phase) > 5:
                from scipy.signal import savgol_filter
                rise_phase_smooth = savgol_filter(rise_phase, min(5, len(rise_phase)), 2)
            else:
                rise_phase_smooth = rise_phase
            
            # Calculate numerical derivative
            slopes = np.diff(rise_phase_smooth) / np.diff(rise_times)
            
            # Find maximum rise slope and its time point
            max_rise_slope = np.max(slopes) if len(slopes) > 0 else 0
            time_of_max_rise = onset_frame + np.argmax(slopes) if len(slopes) > 0 else onset_frame
        else:
            # If rise phase is just one point, can't calculate slope
            max_rise_slope = 0
            time_of_max_rise = onset_frame
        
        logger.info(f"Detected evoked response with max dF/F: {max_value:.4f} at frame {max_frame}")
        
        return {
            'max_value': max_value,
            'max_frame': max_frame,
            'onset_frame': onset_frame,
            'offset_frame': offset_frame,
            'duration': duration,
            'area': area,
            'rise_time': rise_time,
            'max_rise_slope': max_rise_slope,
            'time_of_max_rise': time_of_max_rise,
            'mean_value': mean_value
        }
        
    except Exception as e:
        logger.error(f"Error in evoked response detection: {str(e)}")
        # Return default values on error
        return {
            'max_value': 0.0,
            'max_frame': 0,
            'onset_frame': 0,
            'offset_frame': 0,
            'duration': 0,
            'area': 0.0,
            'rise_time': 0,
            'max_rise_slope': 0.0,
            'time_of_max_rise': 0,
            'mean_value': 0.0
        }

def correct_photobleaching(trace, baseline_frames, logger, condition=None, config=None):
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
    condition : str, optional
        Experimental condition (e.g., "0um", "10um", "25um")
    config : dict, optional
        Configuration parameters
        
    Returns
    -------
    numpy.ndarray
        Photobleaching-corrected trace with flattened baseline
    """
    try:
        # Get photobleaching correction settings
        pb_settings = config.get("photobleaching_correction", {}) if config else {}
        
        # Default extended frames
        default_extended = pb_settings.get("default_extended_frames", [0, 200])
        extended_frames = [default_extended[0], min(default_extended[1], len(trace)-1)]
        
        # Default prominence
        prominence = pb_settings.get("prominence", 0.05)
        
        # Apply condition-specific settings if available
        if condition and "condition_specific" in pb_settings and condition in pb_settings["condition_specific"]:
            condition_config = pb_settings["condition_specific"][condition]
            
            if "extended_frames" in condition_config:
                custom_extended = condition_config["extended_frames"]
                extended_frames = [custom_extended[0], min(custom_extended[1], len(trace)-1)]
                logger.info(f"Using condition-specific range {extended_frames} for {condition} photobleaching correction")
            
            if "prominence" in condition_config:
                prominence = condition_config["prominence"]
                logger.info(f"Using condition-specific prominence {prominence} for {condition} peak detection")
        
        # Get the baseline window based on our extended frames
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
    Calculate baseline using percentile method, excluding both peaks and troughs.
    
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
        
        # Find both positive and negative peaks
        positive_peaks, _ = find_peaks(baseline_window, prominence=0.05)
        negative_peaks, _ = find_peaks(-baseline_window, prominence=0.05)
        
        # Combine and sort peaks
        all_peaks = np.unique(np.concatenate([positive_peaks, negative_peaks]))
        
        # Create mask to exclude peaks and their surrounding frames (±2 frames)
        mask = np.ones(len(baseline_window), dtype=bool)
        for peak in all_peaks:
            start = max(0, peak - 2)
            end = min(len(baseline_window), peak + 3)  # +3 because slicing is exclusive of end
            mask[start:end] = False
        
        # If all frames would be excluded, keep at least half of them
        if not np.any(mask) and len(baseline_window) > 0:
            logger.warning("All baseline frames would be excluded. Using standard percentile calculation.")
            return np.percentile(baseline_window, percentile)
        
        # Calculate percentile of non-peak frames
        if np.sum(mask) > 0:
            filtered_baseline = baseline_window[mask]
            baseline = np.percentile(filtered_baseline, percentile)
            logger.info(f"Calculated baseline using {percentile}th percentile after excluding {len(all_peaks)} peaks/troughs: {baseline:.4f}")
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
    Extract peak parameters from a fluorescence trace, including edge peaks.
    
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

    # New parameters for edge peak detection
    edge_detection = peak_config.get("edge_detection", True)  # Enable edge detection
    edge_threshold = peak_config.get("edge_threshold", 0.03)  # Minimum dF/F value to consider as edge peak
    edge_window = peak_config.get("edge_window", 10)  # Number of frames to check at edges
    edge_rise_threshold = peak_config.get("edge_rise_threshold", 0.005)  # Required slope for edge detection
    
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

        # Check for edge peaks if enabled
        edge_peaks = []
        edge_properties = {"prominences": [], "left_bases": [], "right_bases": [], "widths": []}
        
        if edge_detection:
            # Check start edge
            start_segment = smooth_trace[:edge_window]
            start_max = np.max(start_segment)
            start_max_idx = np.argmax(start_segment)
            
            # Calculate slope at start
            if len(start_segment) > 3:
                start_slope = np.mean(np.diff(start_segment[:3]))
            else:
                start_slope = 0
            
            # Check if there's a significant peak at the start
            if start_max > edge_threshold and start_max_idx == 0 and start_slope < -edge_rise_threshold:
                # This is a downward slope at start (likely continuation of a peak)
                edge_peaks.append(0)
                # Estimate properties for this edge peak
                edge_properties["prominences"].append(start_max - np.min(start_segment))
                edge_properties["left_bases"].append(0)
                # Find where it crosses back below threshold
                for i in range(1, len(start_segment)):
                    if start_segment[i] < edge_threshold:
                        edge_properties["right_bases"].append(i)
                        break
                else:
                    edge_properties["right_bases"].append(edge_window)
                edge_properties["widths"].append(edge_window)
                logger.info(f"Detected start edge peak with value {start_max:.4f}")
            
            # Check end edge
            end_segment = smooth_trace[-edge_window:]
            end_max = np.max(end_segment)
            end_max_idx = np.argmax(end_segment)
            
            # Calculate slope at end
            if len(end_segment) > 3:
                end_slope = np.mean(np.diff(end_segment[-3:]))
            else:
                end_slope = 0
            
            # Check if there's a significant peak at the end
            if end_max > edge_threshold and end_max_idx == len(end_segment)-1 and end_slope > edge_rise_threshold:
                # This is an upward slope at end (likely beginning of a peak)
                edge_peaks.append(len(trace)-1)
                # Estimate properties for this edge peak
                edge_properties["prominences"].append(end_max - np.min(end_segment))
                # Find where it starts rising above threshold
                for i in range(len(end_segment)-2, -1, -1):
                    if end_segment[i] < edge_threshold:
                        edge_properties["left_bases"].append(len(trace) - edge_window + i + 1)
                        break
                else:
                    edge_properties["left_bases"].append(len(trace) - edge_window)
                edge_properties["right_bases"].append(len(trace)-1)
                edge_properties["widths"].append(edge_window)
                logger.info(f"Detected end edge peak with value {end_max:.4f}")
        
        # Combine regular and edge peaks
        if edge_peaks:
            # Convert edge_properties lists to numpy arrays to match find_peaks output
            for key in edge_properties:
                edge_properties[key] = np.array(edge_properties[key])
            
            # Combine peaks and properties
            all_peaks = np.concatenate((peaks, np.array(edge_peaks)))
            # Sort by position
            sort_idx = np.argsort(all_peaks)
            all_peaks = all_peaks[sort_idx]
            
            # Combine and sort properties
            all_properties = {}
            for key in properties:
                if key in edge_properties:
                    combined = np.concatenate((properties[key], edge_properties[key]))
                    all_properties[key] = combined[sort_idx]
                else:
                    # For properties not estimated for edge peaks, use zeros or appropriate values
                    combined = np.concatenate((properties[key], np.zeros(len(edge_peaks))))
                    all_properties[key] = combined[sort_idx]
            
            peaks = all_peaks
            properties = all_properties
            logger.info(f"Added {len(edge_peaks)} edge peaks, total peaks: {len(peaks)}")
            
    except Exception as e:
        logger.warning(f"Error in peak detection: {str(e)}")
        # Return default values if peak detection fails
        return {
            'amplitude': 0.0,
            'time_of_peak': 0,
            'max_rise_slope': 0.0,
            'time_of_max_rise': 0.0,
            'rise_time': 0.0,
            'area_under_curve': 0.0,
            'peak_count': 0,
            'peak_frequency': 0.0
        }
    
    # If no peaks found, return zeros
    if len(peaks) == 0:
        return {
            'amplitude': 0.0,
            'time_of_peak': 0,
            'max_rise_slope': 0.0,
            'time_of_max_rise': 0.0,
            'rise_time': 0.0,
            'area_under_curve': 0.0,
            'peak_count': 0,
            'peak_frequency': 0.0
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
            auc = integrate.simpson(peak_segment, x_range)
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
        'max_amplitude': np.max(smooth_trace[peaks]),           # Added max amplitude across all peaks, previously 'max_amplitude'
        'mean_amplitude': np.mean(smooth_trace[peaks]),    # Added mean amplitude across all peaks, previously 'mean_peak_amplitude'
        'peak_count': len(peaks),                 # Added count of peaks
        'peak_frequency': peak_frequency,         # Added peak frequency
        'time_of_peak': peak_idx,                #previously 'time_of_peak'
        'max_rise_slope': max_rise_slope,
        'time_of_max_rise': time_of_max_rise,
        'rise_time': rise_time,
        'area_under_curve': auc
    }

def extract_spontaneous_activity(
    trace, 
    spont_config, 
    peak_config,  # New parameter
    logger  # Existing parameter
):
    """
    Extract spontaneous activity parameters from baseline fluorescence.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Fluorescence trace during baseline period (dF/F)
    spont_config : dict
        Spontaneous activity detection configuration
    peak_config : dict
        Peak detection configuration
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    dict
        Dictionary of spontaneous activity parameters
    """
    # Use peak detection parameters from the configuration
    prominence = peak_config.get("prominence", 0.03)
    width = peak_config.get("width", 2)
    height = peak_config.get("height", 0.01)
    
    # Find spontaneous peaks
    try:
        peaks, properties = find_peaks(
            trace, 
            prominence=prominence,
            width=width,
            height=height
        )
        
        # Log the number of peaks found
        if len(peaks) > 0:
            logger.info(f"Found {len(peaks)} spontaneous peaks")
        else:
            logger.info("No spontaneous peaks found")
            
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