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
import logging

# Assume logger is configured elsewhere, or add basic config for testing:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Get frame rate for timing calculations
    frame_rate = config.get("frame_rate", 1.67)  # Default to 1.67 Hz if not specified
    
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
            
            # Determine activity based on the configured active_metric
            if active_metric == "spont_peak_frequency":
                # Use the peak frequency calculated by the *main* peak detector (extract_peak_parameters)
                # which now includes edge peaks.
                activity_value = peak_params.get('peak_frequency', 0.0) # Use .get for safety
                is_active = activity_value > active_threshold
                logger.info(f"ROI {i+1} activity determined by main peak frequency (incl. edges): {activity_value:.4f} (metric: {active_metric}, threshold: {active_threshold})")
            elif active_metric == "max_df_f":
                 # Use max_df_f calculated earlier in the loop
                 activity_value = max_df_f
                 is_active = activity_value > active_threshold
                 logger.info(f"ROI {i+1} activity determined by max dF/F: {activity_value:.4f} (metric: {active_metric}, threshold: {active_threshold})")
            else:
                # Fallback or handle other potential metrics if needed
                # Defaulting to use spont_params frequency if metric is unrecognized
                activity_value = spont_params.get('peak_frequency', 0.0)
                is_active = activity_value > active_threshold
                logger.warning(f"Unrecognized active_metric '{active_metric}' for 0um. Defaulting to spont_params frequency.")
                logger.info(f"ROI {i+1} activity determined by default spont_params frequency: {activity_value:.4f} (threshold: {active_threshold})")

            # Add spectral features if enabled
            if config.get("calculate_spectral_features", False):
                # Calculate frequency domain metrics for PCA
                spectral_features = calculate_spectral_features(
                    df_f[analysis_frames[0]:analysis_frames[1]+1],
                    frame_rate,
                    logger
                )
            else:
                spectral_features = {}
            
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

            # Add spectral features if enabled
            if config.get("calculate_spectral_features", False):
                # Calculate frequency domain metrics of baseline
                baseline_spectral = calculate_spectral_features(
                    df_f[baseline_frames[0]:baseline_frames[1]+1],
                    frame_rate,
                    logger
                )
                baseline_spectral = {f'baseline_{k}': v for k, v in baseline_spectral.items()}
                
                # Calculate frequency domain metrics of response
                response_spectral = calculate_spectral_features(
                    df_f[analysis_frames[0]:analysis_frames[1]+1],
                    frame_rate,
                    logger
                )
                response_spectral = {f'response_{k}': v for k, v in response_spectral.items()}
                
                # Combine both
                spectral_features = {**baseline_spectral, **response_spectral}
            else:
                spectral_features = {}
        
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
        
        # Add peak parameters - use separate update to avoid syntax issues
        roi_metrics.update({f'peak_{k}': v for k, v in peak_params.items()})
        
        # Add condition-specific metrics
        if condition == "0um":
            # Spontaneous condition metrics
            roi_metrics.update({
                # Core metrics
                'snr': peak_params.get('snr', 0.0),
                'baseline_variability': peak_params.get('baseline_variability', 0.0),
                
                # Peak shape metrics
                'peak_width': peak_params.get('peak_width', 0.0),
                'decay_time': peak_params.get('decay_time', 0.0),
                'peak_asymmetry': peak_params.get('peak_asymmetry', 0.0),
                
                # Slope metrics
                'avg_rise_slope': peak_params.get('avg_rise_slope', 0.0),
                'max_rise_slope': peak_params.get('max_rise_slope', 0.0),
                'overall_max_slope': peak_params.get('overall_max_slope', 0.0),
                'overall_avg_slope': peak_params.get('overall_avg_slope', 0.0),
                
                # Temporal pattern metrics
                'mean_iei': peak_params.get('mean_iei', 0.0),
                'cv_iei': peak_params.get('cv_iei', 0.0),
                'amplitude_cv': peak_params.get('amplitude_cv', 0.0)
            })
            
            # Add spectral features if available
            if spectral_features:
                roi_metrics.update(spectral_features)
                
        else:
            # Evoked condition metrics
            roi_metrics.update({
                # Response shape metrics
                'evoked_half_width': evoked_params.get('half_width', 0.0),
                'evoked_duration': evoked_params.get('duration', 0.0),
                
                # Timing metrics
                'evoked_latency': evoked_params.get('response_latency', 0),
                'evoked_time_to_peak': evoked_params.get('max_frame', 0),
                
                # Slope metrics
                'evoked_max_rise_slope': evoked_params.get('max_rise_slope', 0.0),
                'evoked_avg_rise_slope': evoked_params.get('avg_rise_slope', 0.0),
                'evoked_time_of_max_rise': evoked_params.get('time_of_max_rise', 0),
                'evoked_decay_slope': evoked_params.get('decay_slope', 0.0),
                
                # Signal quality metrics
                'evoked_reliability': evoked_params.get('response_reliability', 0.0)
            })
            
            # Add spectral features if available
            if spectral_features:
                roi_metrics.update(spectral_features)
        
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
    Extract parameters related to an evoked response curve with enhanced metrics for PCA.
    """
    try:
        # Find the maximum value and its location
        max_value = np.max(trace)
        max_frame = np.argmax(trace)
        
        # If no significant response, return default values
        if max_value <= 0.001:
            return {
                'max_value': 0.0,
                'max_frame': 0,
                'onset_frame': 0,
                'offset_frame': 0,
                'duration': 0,
                'area': 0.0,
                'rise_time': 0,
                'max_rise_slope': 0.0,
                'avg_rise_slope': 0.0,
                'time_of_max_rise': 0,
                'mean_value': 0.0,
                'decay_time': 0,
                'half_width': 0,
                'decay_slope': 0.0,
                'response_reliability': 0.0,
                'response_latency': 0
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
        half_width = duration  # Store half-width as a separate metric
        
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
        
        # Calculate response latency (from start of analysis window to onset)
        response_latency = onset_frame
        
        # Calculate rise slopes
        if len(rise_phase) > 1:
            # Apply Savitzky-Golay filter for smoother derivative calculation
            if len(rise_phase) > 5:
                rise_phase_smooth = savgol_filter(rise_phase, min(5, len(rise_phase)), 2)
            else:
                rise_phase_smooth = rise_phase
            
            # Calculate numerical derivative
            slopes = np.diff(rise_phase_smooth) / np.diff(rise_times)
            
            # Find maximum rise slope and its time point
            max_rise_slope = np.max(slopes) if len(slopes) > 0 else 0
            time_of_max_rise = onset_frame + np.argmax(slopes) if len(slopes) > 0 else onset_frame
            
            # Calculate average rise slope
            avg_rise_slope = np.mean(slopes) if len(slopes) > 0 else 0
        else:
            max_rise_slope = 0
            time_of_max_rise = onset_frame
            avg_rise_slope = 0
        
        # Calculate decay phase metrics
        decay_phase = trace[max_frame:offset_frame+1]
        decay_times = np.arange(len(decay_phase))
        
        # Calculate decay time (frames from peak to half-decay)
        decay_time = len(decay_phase) - 1 if len(decay_phase) > 0 else 0
        
        # Calculate decay slope
        if len(decay_phase) > 1:
            # Apply smoothing
            if len(decay_phase) > 5:
                decay_phase_smooth = savgol_filter(decay_phase, min(5, len(decay_phase)), 2)
            else:
                decay_phase_smooth = decay_phase
                
            # Calculate decay slope
            decay_slopes = np.diff(decay_phase_smooth) / np.diff(decay_times)
            decay_slope = np.mean(decay_slopes) if len(decay_slopes) > 0 else 0
        else:
            decay_slope = 0
        
        # Calculate response reliability based on signal-to-noise ratio
        # Use baseline segment for noise estimation
        try:
            baseline = trace[:onset_frame]
            if len(baseline) > 0:
                baseline_std = np.std(baseline)
                response_reliability = max_value / baseline_std if baseline_std > 0 else 0
            else:
                response_reliability = 0
        except:
            response_reliability = 0
        
        logger.info(f"Detected evoked response with max dF/F: {max_value:.4f} at frame {max_frame}")
        
        return {
            'max_value': max_value,
            'max_frame': max_frame,
            'onset_frame': onset_frame,
            'offset_frame': offset_frame,
            'duration': duration,
            'half_width': half_width,
            'area': area,
            'rise_time': rise_time,
            'max_rise_slope': max_rise_slope,
            'avg_rise_slope': avg_rise_slope,
            'time_of_max_rise': time_of_max_rise,
            'mean_value': mean_value,
            'decay_time': decay_time,
            'decay_slope': decay_slope,
            'response_reliability': response_reliability,
            'response_latency': response_latency
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
            'half_width': 0,
            'area': 0.0,
            'rise_time': 0,
            'max_rise_slope': 0.0,
            'avg_rise_slope': 0.0,
            'time_of_max_rise': 0,
            'mean_value': 0.0,
            'decay_time': 0,
            'decay_slope': 0.0,
            'response_reliability': 0.0,
            'response_latency': 0
        }

def calculate_spectral_features(trace, frame_rate, logger):
    """
    Calculate spectral features from a fluorescence trace for PCA.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Fluorescence trace
    frame_rate : float
        Frame rate in Hz
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    dict
        Dictionary of spectral features
    """
    try:
        from scipy import signal
        
        # Calculate power spectral density
        f, Pxx = signal.welch(trace, fs=frame_rate, nperseg=min(256, len(trace)//2))
        
        # Define frequency bands
        bands = {
            'ultra_low': (0.001, 0.01),  # Ultra low frequency band
            'low': (0.01, 0.1),          # Low frequency band
            'mid': (0.1, 0.5),           # Mid frequency band  
            'high': (0.5, 1.0)           # High frequency band
        }
        
        # Calculate power in each band
        band_powers = {}
        total_power = np.sum(Pxx)
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Find indices for this frequency band
            idx = np.logical_and(f >= low_freq, f <= high_freq)
            
            # Calculate power in this band
            band_power = np.sum(Pxx[idx])
            
            # Normalize by total power
            if total_power > 0:
                band_powers[f'power_{band_name}'] = band_power / total_power
            else:
                band_powers[f'power_{band_name}'] = 0.0
        
        # Calculate spectral centroid (weighted average frequency)
        if np.sum(Pxx) > 0:
            centroid = np.sum(f * Pxx) / np.sum(Pxx)
        else:
            centroid = 0
            
        # Calculate spectral entropy
        psd_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else np.zeros_like(Pxx)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        if len(psd_norm) > 0:
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        else:
            spectral_entropy = 0
            
        # Calculate dominant frequency
        peak_freq_idx = np.argmax(Pxx)
        dominant_freq = f[peak_freq_idx]
        
        # Return all computed features
        return {
            **band_powers,  # Include all band powers
            'spectral_centroid': centroid,
            'spectral_entropy': spectral_entropy,
            'dominant_frequency': dominant_freq
        }
        
    except Exception as e:
        logger.warning(f"Error calculating spectral features: {str(e)}")
        # Return default values
        return {
            'power_ultra_low': 0.0,
            'power_low': 0.0,
            'power_mid': 0.0,
            'power_high': 0.0,
            'spectral_centroid': 0.0,
            'spectral_entropy': 0.0,
            'dominant_frequency': 0.0
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

def extract_peak_parameters(trace, peak_config, logger, baseline_frames=None, baseline=None):
    """
    Extract peak parameters from a fluorescence trace with enhanced metrics for PCA,
    including improved edge detection.
    
    Parameters
    ----------
    trace : numpy.ndarray
        Fluorescence trace (dF/F)
    peak_config : dict
        Peak detection configuration
    logger : logging.Logger
        Logger object
    baseline_frames : list, optional
        Range of frames to use for baseline calculations
    baseline : float, optional
        Pre-calculated baseline value (if available)
        
    Returns
    -------
    dict
        Dictionary of peak parameters
    """
    # Get peak detection parameters
    prominence = peak_config.get("prominence", 0.03)
    width = peak_config.get("width", 1)
    distance = peak_config.get("distance", 5)
    height = peak_config.get("height", 0.03)
    rel_height = peak_config.get("rel_height", 0.8)
    
    # Edge peak detection parameters
    edge_detection = peak_config.get("edge_detection", True)
    # Use the main 'height' as a minimum absolute value for edge peaks too, 
    # or stick to edge_threshold if defined differently. Let's use edge_threshold.
    edge_threshold = peak_config.get("edge_threshold", height) # Default to main height if not specified
    # edge_window is no longer the primary parameter for segment length
    # edge_window = peak_config.get("edge_window", 10) 
    # edge_rise_threshold only used for start edge check now
    edge_rise_threshold = peak_config.get("edge_rise_threshold", 0.005) # Use default if commented out
    edge_peak_frames = peak_config.get("edge_peak_frames", 30) # Frames from edge to analyze
    # edge_std_threshold is no longer used for end-edge detection
    # edge_std_threshold = peak_config.get("edge_std_threshold", 3.0)

    # Check trace length
    if len(trace) < 5: # Need minimum length for processing
        logger.warning("Trace too short for peak detection.")
        # Return default values
        return {
            'amplitude': 0.0, 'time_of_peak': 0, 'max_rise_slope': 0.0, 'time_of_max_rise': 0.0,
            'rise_time': 0.0, 'area_under_curve': 0.0, 'peak_count': 0, 'peak_frequency': 0.0,
            'snr': 0.0, 'baseline_variability': 0.0, 'peak_width': 0.0, 'decay_time': 0.0,
            'peak_asymmetry': 0.0, 'mean_iei': 0.0, 'cv_iei': 0.0, 'amplitude_cv': 0.0,
            'overall_max_slope': 0.0, 'overall_avg_slope': 0.0, 'max_amplitude': 0.0, 'mean_amplitude': 0.0
        }

    # Apply light smoothing
    try:
        # Ensure window size is odd and less than trace length
        smooth_window = 5
        if len(trace) <= smooth_window:
            smooth_window = max(3, len(trace) // 2 * 2 + 1) # Make it odd and smaller
            if smooth_window > len(trace): smooth_window = len(trace) # Max is trace length
        if smooth_window < 3: smooth_window = 3 # Min window size for savgol

        # Check if trace length allows polynomial order 2
        poly_order = 2
        if smooth_window <= poly_order:
             poly_order = smooth_window - 1 # Reduce order if window is too small
             if poly_order < 1: poly_order = 1 # Minimum order is 1

        if len(trace) >= smooth_window and poly_order > 0:
             smooth_trace = savgol_filter(trace, smooth_window, poly_order)
        else:
             smooth_trace = trace # Cannot smooth if trace is too short or window invalid

    except Exception as e:
        logger.warning(f"Smoothing failed: {e}. Using original trace.")
        smooth_trace = trace
            
    # --- Main Peak Detection ---
    try:
        peaks, properties = find_peaks(
            smooth_trace,
            prominence=prominence,
            width=width,
            distance=distance,
            height=height,
            rel_height=rel_height
        )
        logger.info(f"Found {len(peaks)} regular peaks.")
        
    except Exception as e:
        logger.error(f"Error during main peak detection: {e}")
        peaks = np.array([], dtype=int)
        properties = {}

    # --- Edge Peak Detection ---
    edge_peaks_indices = []
    edge_peaks_properties = {} # Store properties for edge peaks separately first

    if edge_detection and len(smooth_trace) > 1:
        # === Check Start Edge ===
        # Looks for a peak that likely started *before* the trace (high value, negative slope at frame 0)
        start_segment_len = min(edge_peak_frames, len(smooth_trace)) # Use consistent frame count
        start_segment = smooth_trace[:start_segment_len]
        if len(start_segment) > 0:
            start_max_idx_rel = np.argmax(start_segment)
            start_max_val = start_segment[start_max_idx_rel]
            
            # Check if the max is at the very beginning and above threshold
            if start_max_idx_rel == 0 and start_max_val >= edge_threshold:
                # Check for negative slope (suggests decay phase of a prior peak)
                if len(start_segment) > 2:
                    start_slope = np.mean(np.diff(start_segment[:3])) # Slope over first few points
                    if start_slope < -abs(edge_rise_threshold): # Use abs() in case config has negative
                        abs_idx = 0
                        if abs_idx not in peaks and abs_idx not in edge_peaks_indices:
                            logger.info(f"Detected start edge peak: val={start_max_val:.4f} at frame {abs_idx} (negative slope {start_slope:.4f})")
                            edge_peaks_indices.append(abs_idx)
                            # Estimate properties crudely
                            edge_peaks_properties.setdefault('prominences', []).append(start_max_val - np.min(start_segment))
                            edge_peaks_properties.setdefault('left_bases', []).append(0) # Cannot know true left base
                            # Estimate right base by finding where it drops below threshold/min
                            try:
                                right_base_rel = next(i for i, v in enumerate(start_segment) if v < edge_threshold / 2 or v < np.min(start_segment) + prominence/2)
                            except StopIteration:
                                right_base_rel = start_segment_len - 1
                            edge_peaks_properties.setdefault('right_bases', []).append(right_base_rel)
                            edge_peaks_properties.setdefault('widths', []).append(right_base_rel) # Crude width


        # === Check End Edge (Revised Logic) ===
        # Looks for any significant peak within the last 'edge_peak_frames'
        end_segment_len = min(edge_peak_frames, len(smooth_trace))
        end_segment = smooth_trace[-end_segment_len:]

        if len(end_segment) >= max(5, width + 1): # Need sufficient length for find_peaks
            try:
                # Run find_peaks on the end segment using MAIN parameters
                segment_peaks_rel, segment_properties = find_peaks(
                    end_segment,
                    prominence=prominence, # Main prominence
                    width=width,           # Main width
                    height=height,         # Main height requirement
                    distance=1             # Allow peaks close together near edge
                )

                # Iterate through peaks found in the segment
                for i, rel_idx in enumerate(segment_peaks_rel):
                    peak_val = end_segment[rel_idx]
                    # Check if it also meets the specific edge_threshold (can be same as height or different)
                    if peak_val >= edge_threshold:
                        abs_idx = len(smooth_trace) - end_segment_len + rel_idx
                        
                        # Add if it's not already found by global search or previous edge search
                        if abs_idx not in peaks and abs_idx not in edge_peaks_indices:
                            logger.info(f"Detected end edge peak via segment analysis: val={peak_val:.4f} at frame {abs_idx}")
                            edge_peaks_indices.append(abs_idx)
                            
                            # Add properties from segment analysis, adjusting bases to absolute
                            for key, values in segment_properties.items():
                                prop_val = values[i]
                                # Adjust base indices relative -> absolute
                                if key in ['left_bases', 'right_bases']:
                                     # Ensure integer index before adding offset
                                     abs_base_idx = len(smooth_trace) - end_segment_len + int(np.round(prop_val))
                                     # Clamp right base to trace end
                                     if key == 'right_bases':
                                         abs_base_idx = min(abs_base_idx, len(smooth_trace) - 1)
                                     # Clamp left base to trace start
                                     if key == 'left_bases':
                                          abs_base_idx = max(0, abs_base_idx)
                                     prop_val = abs_base_idx
                                
                                edge_peaks_properties.setdefault(key, []).append(prop_val)
                            
                            # Ensure essential properties exist if find_peaks didn't provide them
                            if 'prominences' not in edge_peaks_properties or len(edge_peaks_properties['prominences']) < len(edge_peaks_indices):
                                edge_peaks_properties.setdefault('prominences', []).append(peak_val - np.min(end_segment))
                            if 'widths' not in edge_peaks_properties or len(edge_peaks_properties['widths']) < len(edge_peaks_indices):
                                edge_peaks_properties.setdefault('widths', []).append(width) # Default width
                            if 'left_bases' not in edge_peaks_properties or len(edge_peaks_properties['left_bases']) < len(edge_peaks_indices):
                                # Estimate left base based on width
                                edge_peaks_properties.setdefault('left_bases', []).append(max(0, abs_idx - int(width/2)))
                            if 'right_bases' not in edge_peaks_properties or len(edge_peaks_properties['right_bases']) < len(edge_peaks_indices):
                                # Estimate right base based on width, clamped to end
                                edge_peaks_properties.setdefault('right_bases', []).append(min(len(smooth_trace) - 1, abs_idx + int(width/2)))


            except Exception as e:
                logger.warning(f"Error during end segment peak detection: {e}")

    # --- Combine Regular and Edge Peaks ---
    if edge_peaks_indices:
        logger.info(f"Combining {len(peaks)} regular peaks with {len(edge_peaks_indices)} edge peaks.")
        
        # Convert edge_peaks_properties values to numpy arrays for concatenation
        for key in edge_peaks_properties:
            edge_peaks_properties[key] = np.array(edge_peaks_properties[key])
            
        all_peak_indices = np.concatenate((peaks, np.array(edge_peaks_indices, dtype=int)))
        
        # Combine properties, filling missing ones for edge peaks if necessary
        all_properties = {}
        # Get all keys from both dictionaries
        all_keys = set(properties.keys()) | set(edge_peaks_properties.keys())

        for key in all_keys:
            props_regular = properties.get(key, np.zeros(len(peaks))) # Default to zeros if key missing
            props_edge = edge_peaks_properties.get(key, np.zeros(len(edge_peaks_indices))) # Default to zeros

            # Ensure arrays are compatible (e.g., same dtype if possible) before concatenating
            if props_regular.dtype != props_edge.dtype:
                 # Attempt to cast edge props to regular props dtype, or use a common type like float
                 try:
                     props_edge = props_edge.astype(props_regular.dtype)
                 except ValueError:
                     # Fallback to float64 if direct casting fails
                     common_dtype = np.float64
                     props_regular = props_regular.astype(common_dtype)
                     props_edge = props_edge.astype(common_dtype)

            combined = np.concatenate((props_regular, props_edge))
            all_properties[key] = combined
            
        # Sort peaks by index
        sort_idx = np.argsort(all_peak_indices)
        peaks = all_peak_indices[sort_idx]
        
        # Sort all properties accordingly
        for key in all_properties:
            all_properties[key] = all_properties[key][sort_idx]
            
        properties = all_properties # Use the combined and sorted properties
        logger.info(f"Total peaks after combination and sorting: {len(peaks)}")

    # --- Calculate Metrics Based on Combined Peaks ---
    if len(peaks) == 0:
        logger.info("No peaks found after edge detection.")
        # Return default values
        return {
            'amplitude': 0.0, 'time_of_peak': 0, 'max_rise_slope': 0.0, 'time_of_max_rise': 0.0,
            'rise_time': 0.0, 'area_under_curve': 0.0, 'peak_count': 0, 'peak_frequency': 0.0,
            'snr': 0.0, 'baseline_variability': 0.0, 'peak_width': 0.0, 'decay_time': 0.0,
            'peak_asymmetry': 0.0, 'mean_iei': 0.0, 'cv_iei': 0.0, 'amplitude_cv': 0.0,
            'overall_max_slope': 0.0, 'overall_avg_slope': 0.0, 'max_amplitude': 0.0, 'mean_amplitude': 0.0
        }

    # --- Proceed with metric calculations using the final 'peaks' and 'properties' ---
    
    # Find the most prominent peak among *all* detected peaks
    if 'prominences' in properties and len(properties['prominences']) > 0:
        max_prominence_idx_in_list = np.argmax(properties['prominences'])
    else:
        # Fallback if prominence is missing (shouldn't happen often with revised logic)
        max_prominence_idx_in_list = np.argmax(smooth_trace[peaks]) if len(peaks) > 0 else 0

    if max_prominence_idx_in_list >= len(peaks): # Safety check
        logger.warning("Index calculation error for max prominence peak. Defaulting to first peak.")
        max_prominence_idx_in_list = 0
        
    peak_idx = peaks[max_prominence_idx_in_list] # Absolute frame index of the most prominent peak
    
    # Get peak amplitude (using the value from the smoothed trace at the peak index)
    peak_amplitude = smooth_trace[peak_idx]
    
    # Get time of peak (frame number)
    time_of_peak = peak_idx

    # --- Calculate derived metrics (SNR, IEI, CV, Slopes, AUC, Widths etc.) ---
    # (This part of the code seems okay, but ensure it uses the potentially updated 'properties')

    # Baseline calculations for SNR and variability
    snr = 0.0
    baseline_variability = 0.0
    if baseline_frames is not None and len(baseline_frames) == 2:
         baseline_start, baseline_end = baseline_frames
         baseline_start = max(0, baseline_start)
         baseline_end = min(len(trace) -1, baseline_end)
         if baseline_end > baseline_start:
             baseline_segment = trace[baseline_start:baseline_end+1]
             baseline_std = np.std(baseline_segment)
             baseline_mean = np.mean(baseline_segment)
             if baseline_std > 1e-9: # Avoid division by zero or near-zero std
                 snr = peak_amplitude / baseline_std
             if abs(baseline_mean) > 1e-9: # Avoid division by zero baseline mean
                 baseline_variability = baseline_std / abs(baseline_mean)

    # Inter-event intervals (IEI)
    mean_iei = 0.0
    cv_iei = 0.0
    if len(peaks) > 1:
        iei = np.diff(peaks)
        if len(iei) > 0:
             mean_iei = np.mean(iei)
             if mean_iei > 1e-9:
                 cv_iei = np.std(iei) / mean_iei

    # Amplitude Coefficient of Variation (CV)
    amplitude_cv = 0.0
    all_peak_amplitudes = smooth_trace[peaks]
    if len(all_peak_amplitudes) > 1:
        mean_amp = np.mean(all_peak_amplitudes)
        if abs(mean_amp) > 1e-9:
            amplitude_cv = np.std(all_peak_amplitudes) / abs(mean_amp)
            
    # Max and Mean Amplitude across all detected peaks
    max_amplitude = np.max(all_peak_amplitudes) if len(all_peak_amplitudes) > 0 else 0.0
    mean_peak_amplitude = np.mean(all_peak_amplitudes) if len(all_peak_amplitudes) > 0 else 0.0

    # Peak frequency (per 100 frames)
    peak_frequency = len(peaks) / (len(trace) / 100.0) if len(trace) > 0 else 0.0

    # Metrics related to the *most prominent* peak (using max_prominence_idx_in_list)
    max_rise_slope = 0.0
    avg_rise_slope = 0.0
    time_of_max_rise = 0
    rise_time = 0.0
    auc = 0.0
    peak_width = 0.0
    decay_time = 0.0
    peak_asymmetry = 0.0

    try:
        # Get bases using the index of the most prominent peak
        left_base = int(np.round(properties['left_bases'][max_prominence_idx_in_list])) if 'left_bases' in properties else max(0, peak_idx - int(width / 2))
        right_base = int(np.round(properties['right_bases'][max_prominence_idx_in_list])) if 'right_bases' in properties else min(len(trace)-1, peak_idx + int(width / 2))
        
        # Ensure bases are within trace bounds and left < right
        left_base = max(0, left_base)
        right_base = min(len(trace) - 1, right_base)
        if left_base >= peak_idx: left_base = max(0, peak_idx - 1) # Ensure rise phase exists
        if right_base <= peak_idx: right_base = min(len(trace) - 1, peak_idx + 1) # Ensure decay phase exists
        if left_base >= right_base: # Handle case where bases are invalid
             if left_base > 0 : left_base = left_base -1
             else: right_base = right_base + 1


        # Rise Phase calculations
        if peak_idx > left_base:
            rise_phase = smooth_trace[left_base:peak_idx+1]
            rise_times = np.arange(len(rise_phase))
            if len(rise_phase) > 1:
                # Use savgol filter on the phase if long enough
                rise_smooth_window = min(5, len(rise_phase))
                if rise_smooth_window % 2 == 0: rise_smooth_window -= 1 # Must be odd
                rise_poly_order = min(2, rise_smooth_window -1) if rise_smooth_window > 1 else 0

                if rise_smooth_window >=3 and rise_poly_order >= 1:
                     rise_phase_smooth = savgol_filter(rise_phase, rise_smooth_window, rise_poly_order)
                else:
                     rise_phase_smooth = rise_phase

                slopes = np.diff(rise_phase_smooth) / np.diff(rise_times) if len(rise_times) > 1 else np.array([0])
                if len(slopes) > 0:
                     max_rise_slope = np.max(slopes)
                     avg_rise_slope = np.mean(slopes)
                     time_of_max_rise = left_base + np.argmax(slopes) # Frame index of max slope

                # Rise time (10% to 90%)
                try:
                    amp_10 = smooth_trace[left_base] + 0.1 * (peak_amplitude - smooth_trace[left_base])
                    amp_90 = smooth_trace[left_base] + 0.9 * (peak_amplitude - smooth_trace[left_base])
                    rise_start_idx = left_base + next((i for i, v in enumerate(rise_phase) if v >= amp_10), 0)
                    rise_end_idx = left_base + next((i for i, v in enumerate(rise_phase) if v >= amp_90), len(rise_phase)-1)
                    rise_time = rise_end_idx - rise_start_idx
                except StopIteration:
                    rise_time = peak_idx - left_base # Fallback: time from base to peak
        
        # Area Under Curve (AUC) for the main peak
        if right_base > left_base:
             x_range = np.arange(left_base, right_base+1)
             peak_segment = smooth_trace[left_base:right_base+1]
             # Subtract baseline estimated at left_base for AUC calculation? Optional.
             # peak_segment_baseline_corr = peak_segment - smooth_trace[left_base]
             # auc = integrate.simpson(peak_segment_baseline_corr, x_range)
             # Or just integrate the segment directly:
             auc = integrate.simpson(peak_segment, x_range)

        # Width and Decay calculations
        peak_width = properties.get('widths', [0])[max_prominence_idx_in_list] if 'widths' in properties else 0
        
        # Decay time (peak to 50% decay relative to baseline at left base)
        if peak_idx < right_base:
            decay_phase = smooth_trace[peak_idx:right_base+1]
            baseline_val = smooth_trace[left_base] # Use start of rise as baseline reference
            half_decay_val = baseline_val + (peak_amplitude - baseline_val) / 2.0
            
            decay_time = 0
            try:
                decay_idx_rel = next((i for i, v in enumerate(decay_phase) if v <= half_decay_val), -1)
                if decay_idx_rel != -1:
                    decay_time = decay_idx_rel # Frames from peak
            except StopIteration:
                 decay_time = right_base - peak_idx # Fallback: time to end of identified base

        # Peak Asymmetry
        if rise_time > 0 and decay_time > 0:
            peak_asymmetry = decay_time / rise_time
            
    except Exception as e:
        logger.warning(f"Error calculating metrics for the most prominent peak ({peak_idx}): {e}")
        # Reset potentially calculated values to zero if error occurs mid-calculation
        max_rise_slope, avg_rise_slope, time_of_max_rise, rise_time = 0.0, 0.0, 0, 0.0
        auc, peak_width, decay_time, peak_asymmetry = 0.0, 0.0, 0.0, 0.0


    # Overall slope metrics (calculated across *all* peaks' rise phases)
    overall_max_slope = 0.0
    overall_avg_slope = 0.0
    all_rise_slopes = []
    try:
        for i, p_idx in enumerate(peaks):
            lb = int(np.round(properties['left_bases'][i])) if 'left_bases' in properties else max(0, p_idx - int(width / 2))
            lb = max(0, lb)
            if p_idx > lb:
                 p_rise_phase = smooth_trace[lb:p_idx+1]
                 if len(p_rise_phase) > 1:
                     p_times = np.arange(len(p_rise_phase))
                     p_slopes = np.diff(p_rise_phase) / np.diff(p_times)
                     all_rise_slopes.extend(p_slopes)
        
        if all_rise_slopes:
            overall_max_slope = np.max(all_rise_slopes)
            overall_avg_slope = np.mean(all_rise_slopes)
        else: # Fallback if no slopes calculated
             overall_max_slope = max_rise_slope
             overall_avg_slope = avg_rise_slope

    except Exception as e:
        logger.warning(f"Error calculating overall slope metrics: {e}")
        overall_max_slope = max_rise_slope # Fallback
        overall_avg_slope = avg_rise_slope # Fallback

    # Return combined dictionary
    return {
        # --- Metrics for the most prominent peak ---
        'amplitude': peak_amplitude, # Amplitude of the most prominent peak
        'time_of_peak': time_of_peak, # Frame index of the most prominent peak
        'max_rise_slope': max_rise_slope, # Max slope during rise of most prominent peak
        'avg_rise_slope': avg_rise_slope, # Average slope during rise of most prominent peak
        'time_of_max_rise': time_of_max_rise, # Frame index of max slope for most prominent peak
        'rise_time': rise_time, # Rise time (e.g., 10-90%) of most prominent peak
        'area_under_curve': auc, # Area under the curve of the most prominent peak
        'peak_width': peak_width, # Width (e.g., at half-prominence) of most prominent peak
        'decay_time': decay_time, # Decay time (e.g., peak to 50%) of most prominent peak
        'peak_asymmetry': peak_asymmetry, # Ratio of decay_time / rise_time for most prominent peak

        # --- Metrics across all detected peaks ---
        'peak_count': len(peaks), # Total number of detected peaks (regular + edge)
        'peak_frequency': peak_frequency, # Total peaks per 100 frames
        'max_amplitude': max_amplitude, # Maximum amplitude across all detected peaks
        'mean_amplitude': mean_peak_amplitude, # Mean amplitude across all detected peaks
        'mean_iei': mean_iei, # Mean inter-event interval
        'cv_iei': cv_iei, # Coefficient of variation for IEI
        'amplitude_cv': amplitude_cv, # Coefficient of variation for peak amplitudes
        'overall_max_slope': overall_max_slope, # Maximum slope found across all peak rises
        'overall_avg_slope': overall_avg_slope, # Average slope across all peak rises
        
        # --- Signal Quality Metrics ---
        'snr': snr, # Signal-to-noise ratio (prominent peak amp / baseline std)
        'baseline_variability': baseline_variability, # Baseline std / baseline mean
    }
    
    # Find the most prominent peak
    max_peak_idx = np.argmax(properties['prominences'])
    peak_idx = peaks[max_peak_idx]
    
    # Get peak amplitude
    peak_amplitude = smooth_trace[peak_idx]
    
    # Get time of peak (frame number)
    time_of_peak = peak_idx
    
    # Calculate signal-to-noise ratio and baseline variability
    if baseline_frames is not None:
        baseline_segment = trace[baseline_frames[0]:baseline_frames[1]+1]
        baseline_std = np.std(baseline_segment)
        baseline_mean = np.mean(baseline_segment)
        snr = peak_amplitude / baseline_std if baseline_std > 0 else 0
        baseline_variability = baseline_std / baseline_mean if baseline_mean > 0 else 0
    else:
        snr = 0
        baseline_variability = 0
    
    # Calculate interevent intervals if multiple peaks
    if len(peaks) > 1:
        iei = np.diff(peaks)
        mean_iei = np.mean(iei)
        cv_iei = np.std(iei) / mean_iei if mean_iei > 0 else 0
    else:
        mean_iei = 0
        cv_iei = 0
    
    # Calculate peak amplitude variations
    if len(peaks) > 1:
        peak_amplitudes = smooth_trace[peaks]
        amplitude_cv = np.std(peak_amplitudes) / np.mean(peak_amplitudes) if np.mean(peak_amplitudes) > 0 else 0
    else:
        amplitude_cv = 0
    
    # Calculate rise slope metrics
    try:
        left_base = int(properties['left_bases'][max_peak_idx])
        
        # Ensure valid range
        if left_base == peak_idx:
            left_base = max(0, peak_idx - 1)
        
        # Calculate slope for each point during rise phase
        rise_phase = smooth_trace[left_base:peak_idx+1]
        rise_times = np.arange(len(rise_phase))
        
        # Apply Savitzky-Golay filter for smoother derivative calculation
        if len(rise_phase) > 5:
            rise_phase_smooth = savgol_filter(rise_phase, min(5, len(rise_phase)), 2)
        else:
            rise_phase_smooth = rise_phase
        
        # Calculate numerical derivative
        slopes = np.diff(rise_phase_smooth) / np.diff(rise_times) if len(rise_times) > 1 else [0]
        
        # Find maximum rise slope
        max_rise_slope = np.max(slopes) if len(slopes) > 0 else 0
        time_of_max_rise = left_base + np.argmax(slopes) if len(slopes) > 0 else 0
        # Calculate average rise slope
        avg_rise_slope = np.mean(slopes) if len(slopes) > 0 else 0
        
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
        avg_rise_slope = 0
        rise_time = 0
        auc = 0
    
    # Calculate width at half maximum and decay time
    try:
        # Calculate peak width using scipy's width function if available
        peak_width = properties.get('widths', [0])[max_peak_idx] if max_peak_idx < len(properties.get('widths', [])) else 0
        
        # Calculate decay time (time from peak to 50% decay)
        right_base = int(properties['right_bases'][max_peak_idx]) if max_peak_idx < len(properties.get('right_bases', [])) else peak_idx + 1
        decay_phase = smooth_trace[peak_idx:right_base+1]
        half_decay_val = peak_amplitude - (peak_amplitude - baseline) / 2 if baseline is not None else peak_amplitude / 2
        
        decay_time = 0
        for i, v in enumerate(decay_phase[1:], 1):
            if v <= half_decay_val:
                decay_time = i
                break
        
        # Calculate peak asymmetry (rise time vs decay time)
        if rise_time > 0 and decay_time > 0:
            peak_asymmetry = decay_time / rise_time
        else:
            peak_asymmetry = 0
    except Exception as e:
        logger.warning(f"Error calculating width/decay metrics: {str(e)}")
        peak_width = 0
        decay_time = 0
        peak_asymmetry = 0
    
    # Calculate metrics across all peaks for better characterization
    try:
        # Calculate slope metrics for all peaks
        all_slopes = []
        
        for i, p in enumerate(peaks):
            try:
                left_idx = int(properties['left_bases'][i]) if i < len(properties.get('left_bases', [])) else max(0, p-1)
                # Get the rise phase for this peak
                if left_idx < p:
                    p_rise = smooth_trace[left_idx:p+1]
                    if len(p_rise) > 1:
                        p_times = np.arange(len(p_rise))
                        p_slopes = np.diff(p_rise) / np.diff(p_times)
                        all_slopes.extend(p_slopes)
            except:
                pass
        
        # Overall statistics across all peaks
        overall_max_slope = np.max(all_slopes) if all_slopes else max_rise_slope
        overall_avg_slope = np.mean(all_slopes) if all_slopes else avg_rise_slope
        
    except Exception as e:
        logger.warning(f"Error calculating overall slope metrics: {str(e)}")
        overall_max_slope = max_rise_slope
        overall_avg_slope = avg_rise_slope
    
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
        'max_amplitude': max_amplitude,
        'mean_amplitude': mean_peak_amplitude,
        'peak_count': len(peaks),
        'peak_frequency': peak_frequency,
        'time_of_peak': time_of_peak,
        'max_rise_slope': max_rise_slope,
        'avg_rise_slope': avg_rise_slope,
        'time_of_max_rise': time_of_max_rise,
        'rise_time': rise_time,
        'area_under_curve': auc,
        'snr': snr,
        'baseline_variability': baseline_variability,
        'peak_width': peak_width,
        'decay_time': decay_time,
        'peak_asymmetry': peak_asymmetry,
        'mean_iei': mean_iei,
        'cv_iei': cv_iei,
        'amplitude_cv': amplitude_cv,
        'overall_max_slope': overall_max_slope,
        'overall_avg_slope': overall_avg_slope
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