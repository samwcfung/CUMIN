# modules/visualization_helpers.py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive, fixed
import cv2
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import find_peaks

def normalize_for_display(img):
    """Normalize image for display"""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def create_denoising_visualization(image_data, logger, config):
    """Create interactive visualization for Gaussian denoising"""
    try:
        # Frame selector
        frame_idx_slider = widgets.IntSlider(
            value=10,
            min=0,
            max=min(image_data.shape[0]-1, 100),
            step=1,
            description='Frame:',
            style={'description_width': 'initial'}
        )
        
        # Denoising parameters
        ksize_slider = widgets.IntSlider(
            value=5,
            min=1,
            max=21,
            step=2,
            description='Kernel Size:',
            style={'description_width': 'initial'}
        )
        
        sigmaX_slider = widgets.FloatSlider(
            value=1.5,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Sigma X:',
            style={'description_width': 'initial'}
        )
        
        # Define the display function
        def display_denoising(frame_idx, ksize, sigmaX):
            # Ensure ksize is odd
            if ksize % 2 == 0:
                ksize += 1
                
            # Get the selected frame
            sample_frame = image_data[frame_idx].copy()
            
            # Apply Gaussian denoising
            denoised_frame = cv2.GaussianBlur(sample_frame, (ksize, ksize), sigmaX)
            
            # Display original vs denoised
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Display original frame
            axes[0].imshow(normalize_for_display(sample_frame), cmap='gray')
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            
            # Display denoised frame
            axes[1].imshow(normalize_for_display(denoised_frame), cmap='gray')
            axes[1].set_title(f'Gaussian Denoised (k={ksize}, σ={sigmaX:.1f})')
            axes[1].axis('off')
            
            # Display difference
            diff = np.abs(sample_frame - denoised_frame)
            axes[2].imshow(normalize_for_display(diff), cmap='hot')
            axes[2].set_title('Difference (Red=More Change)')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Update config with current values
            if 'denoise' not in config['preprocessing']:
                config['preprocessing']['denoise'] = {}
            
            config['preprocessing']['denoise']['enabled'] = True
            config['preprocessing']['denoise']['method'] = 'gaussian'
            config['preprocessing']['denoise']['params'] = {
                'ksize': [ksize, ksize],
                'sigmaX': sigmaX
            }
            
            print(f"Parameters updated: ksize={ksize}, sigmaX={sigmaX}")
        
        # Create and return interactive widget
        return interactive(
            display_denoising,
            frame_idx=frame_idx_slider,
            ksize=ksize_slider,
            sigmaX=sigmaX_slider
        )
        
    except Exception as e:
        print(f"Error creating denoising visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_pnr_refinement_visualization(roi_data, roi_masks, logger, config):
    """Create interactive visualization for PNR refinement"""
    try:
        # Import needed modules here for self-containment
        from modules.roi_processing import split_signal_noise
        
        # Create widgets for PNR refinement parameters
        noise_freq_cutoff_slider = widgets.FloatSlider(
            value=0.03,
            min=0.01,
            max=0.2,
            step=0.01,
            description='Noise Freq Cutoff:',
            style={'description_width': 'initial'}
        )
        
        percentile_threshold_slider = widgets.FloatSlider(
            value=99,
            min=90,
            max=99.9,
            step=0.1,
            description='Percentile Threshold:',
            style={'description_width': 'initial'}
        )
        
        trace_smoothing_slider = widgets.IntSlider(
            value=3,
            min=0,
            max=15,
            step=1,
            description='Trace Smoothing:',
            style={'description_width': 'initial'}
        )
        
        min_pnr_slider = widgets.FloatSlider(
            value=8.0,
            min=3.0,
            max=20.0,
            step=0.5,
            description='Min PNR:',
            style={'description_width': 'initial'}
        )
        
        # Widget to select ROIs to display
        roi_selector = widgets.SelectMultiple(
            options=[(f"ROI {i+1}", i) for i in range(min(10, len(roi_data)))],
            value=[0, 1, 2],  # Default: first 3 ROIs
            description='ROIs to Display:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        # Define the display function
        def display_pnr_refinement(noise_freq_cutoff, percentile_threshold, trace_smoothing, min_pnr, roi_indices):
            if not roi_indices:
                print("Please select at least one ROI to display")
                return
                
            # Split traces into signal and noise components
            sample_traces = roi_data[list(roi_indices)]
            signal_traces, noise_traces = split_signal_noise(sample_traces, noise_freq_cutoff, logger)
            
            # Apply smoothing if enabled
            if trace_smoothing > 0:
                smoothed_signal = np.zeros_like(signal_traces)
                for i in range(len(signal_traces)):
                    window = np.ones(trace_smoothing) / trace_smoothing
                    smoothed_signal[i] = np.convolve(signal_traces[i], window, mode='same')
            else:
                smoothed_signal = signal_traces
            
            # Compute PNR values
            pnr_values = np.zeros(len(roi_indices))
            for i in range(len(roi_indices)):
                # Get peak value (using percentile)
                peak_value = np.percentile(smoothed_signal[i], percentile_threshold)
                
                # Calculate noise standard deviation
                noise_std = np.std(noise_traces[i])
                
                # Avoid division by zero
                if noise_std > 0:
                    pnr_values[i] = peak_value / noise_std
                else:
                    pnr_values[i] = 0
            
            # Display traces and PNR values
            n_rois = len(roi_indices)
            fig, axes = plt.subplots(n_rois, 3, figsize=(15, 4*n_rois))
            
            # Handle single ROI case
            if n_rois == 1:
                axes = np.array([axes])
            
            for i, roi_idx in enumerate(roi_indices):
                # Original trace
                axes[i, 0].plot(roi_data[roi_idx], 'k-', label=f'Original')
                axes[i, 0].set_title(f'ROI {roi_idx+1} - Original Trace')
                axes[i, 0].set_xlabel('Frame')
                axes[i, 0].set_ylabel('Fluorescence')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Signal component
                axes[i, 1].plot(signal_traces[i], 'g-', label='Signal')
                if trace_smoothing > 0:
                    axes[i, 1].plot(smoothed_signal[i], 'r-', label='Smoothed Signal')
                axes[i, 1].set_title(f'Signal Component (cutoff={noise_freq_cutoff})')
                axes[i, 1].set_xlabel('Frame')
                axes[i, 1].set_ylabel('Fluorescence')
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].legend()
                
                # Noise component
                axes[i, 2].plot(noise_traces[i], 'b-', label='Noise')
                axes[i, 2].set_title(f'Noise Component (PNR={pnr_values[i]:.2f})')
                axes[i, 2].set_xlabel('Frame')
                axes[i, 2].set_ylabel('Fluorescence')
                axes[i, 2].grid(True, alpha=0.3)
                
                # Add PNR threshold line and indication if the ROI passes the threshold
                axes[i, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                if pnr_values[i] >= min_pnr:
                    status = "PASS"
                    color = 'green'
                else:
                    status = "FAIL"
                    color = 'red'
                
                axes[i, 2].text(0.05, 0.95, f"PNR: {pnr_values[i]:.2f} ({status})", 
                                transform=axes[i, 2].transAxes, 
                                fontsize=10, va='top', ha='left',
                                bbox=dict(facecolor=color, alpha=0.3))
            
            plt.tight_layout()
            plt.show()
            
            # Display summary
            n_pass = sum(pnr >= min_pnr for pnr in pnr_values)
            print(f"PNR Summary: {n_pass}/{len(roi_indices)} selected ROIs pass the threshold (>= {min_pnr})")
            
            # Update config with current values (for reference)
            if 'pnr_refinement' not in config['roi_processing']:
                config['roi_processing']['pnr_refinement'] = {}
            
            config['roi_processing']['pnr_refinement']['noise_freq_cutoff'] = noise_freq_cutoff
            config['roi_processing']['pnr_refinement']['min_pnr'] = min_pnr
            config['roi_processing']['pnr_refinement']['percentile_threshold'] = percentile_threshold
            config['roi_processing']['pnr_refinement']['trace_smoothing'] = trace_smoothing
            
            print(f"Updated config with: noise_freq_cutoff={noise_freq_cutoff}, min_pnr={min_pnr}")
            print(f"percentile_threshold={percentile_threshold}, trace_smoothing={trace_smoothing}")
            print("To apply these settings to your pipeline, update your config.yaml file.")
        
        # Create and return interactive widget
        return interactive(
            display_pnr_refinement,
            noise_freq_cutoff=noise_freq_cutoff_slider,
            percentile_threshold=percentile_threshold_slider,
            trace_smoothing=trace_smoothing_slider,
            min_pnr=min_pnr_slider,
            roi_indices=roi_selector
        )
        
    except Exception as e:
        print(f"Error creating PNR refinement visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_background_subtraction_visualization(image_data, roi_data, roi_masks, logger, config):
    """Create interactive visualization for background subtraction"""
    try:
        # Import needed modules here for self-containment
        from modules.roi_processing import subtract_background, subtract_global_background
        
        # Create widgets for background subtraction parameters
        bg_method_selector = widgets.Dropdown(
            options=[
                ('Darkest Pixels', 'darkest_pixels'), 
                ('ROI Periphery', 'roi_periphery'),
                ('Global Background', 'global_background')
            ],
            value='darkest_pixels',
            description='Method:',
            style={'description_width': 'initial'}
        )
        
        percentile_slider = widgets.FloatSlider(
            value=0.2,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Percentile (%):',
            style={'description_width': 'initial'}
        )
        
        min_bg_area_slider = widgets.IntSlider(
            value=200,
            min=50,
            max=1000,
            step=50,
            description='Min Background Area:',
            style={'description_width': 'initial'}
        )
        
        median_filter_slider = widgets.IntSlider(
            value=5,
            min=0,
            max=15,
            step=2,
            description='Median Filter Size:',
            style={'description_width': 'initial'}
        )
        
        periphery_size_slider = widgets.IntSlider(
            value=2,
            min=1,
            max=10,
            step=1,
            description='Periphery Size:',
            style={'description_width': 'initial'}
        )
        
        # Widget to select ROIs to display
        roi_selector = widgets.SelectMultiple(
            options=[(f"ROI {i+1}", i) for i in range(min(10, len(roi_data)))],
            value=[0, 1, 2],  # Default: first 3 ROIs
            description='ROIs to Display:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        # Define the display function
        def display_background_subtraction(bg_method, percentile, min_bg_area, median_filter_size, periphery_size, roi_indices):
            if not roi_indices:
                print("Please select at least one ROI to display")
                return
            
            # Create configuration for background subtraction
            bg_config = {
                "method": bg_method,
                "percentile": percentile,
                "min_background_area": min_bg_area,
                "median_filter_size": median_filter_size,
                "periphery_size": periphery_size
            }
            
            # Get ROI data for selected ROIs
            selected_roi_data = roi_data[list(roi_indices)]
            selected_roi_masks = [roi_masks[i] for i in roi_indices]
            
            # Apply background subtraction
            if bg_method == 'global_background':
                bg_corrected_data = subtract_global_background(
                    image_data, 
                    selected_roi_data,
                    selected_roi_masks,
                    bg_config,
                    logger
                )
            else:
                bg_corrected_data = subtract_background(
                    image_data, 
                    selected_roi_data,
                    selected_roi_masks,
                    bg_config,
                    logger
                )
            
            # Display original vs background-corrected traces
            n_rois = len(roi_indices)
            fig, axes = plt.subplots(n_rois, 2, figsize=(15, 4*n_rois))
            
            # Handle single ROI case
            if n_rois == 1:
                axes = np.array([axes])
            
            for i, roi_idx in enumerate(roi_indices):
                # Original trace
                axes[i, 0].plot(selected_roi_data[i], 'k-', label=f'Original')
                axes[i, 0].set_title(f'ROI {roi_idx+1} - Original Trace')
                axes[i, 0].set_xlabel('Frame')
                axes[i, 0].set_ylabel('Fluorescence')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Background-corrected trace
                axes[i, 1].plot(bg_corrected_data[i], 'g-', label='Background Corrected')
                axes[i, 1].set_title(f'Background Corrected ({bg_method})')
                axes[i, 1].set_xlabel('Frame')
                axes[i, 1].set_ylabel('Fluorescence')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Add zero line for reference
                axes[i, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Visualize background mask (for darkest_pixels method)
            if bg_method == 'darkest_pixels':
                # Create darkest pixels mask
                avg_intensity = np.mean(image_data, axis=0)
                threshold = np.percentile(avg_intensity, percentile)
                darkest_pixels_mask = avg_intensity <= threshold
                
                # Apply median filter to remove noise
                if median_filter_size > 0:
                    darkest_pixels_mask = median_filter(darkest_pixels_mask.astype(float), 
                                                       size=median_filter_size) > 0.5
                
                # Create a visualization of the background mask
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Display average intensity image
                axes[0].imshow(normalize_for_display(avg_intensity), cmap='gray')
                axes[0].set_title('Average Intensity')
                axes[0].axis('off')
                
                # Display background mask
                axes[1].imshow(darkest_pixels_mask, cmap='hot')
                axes[1].set_title(f'Background Mask (percentile={percentile}%)')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # Visualize ROI periphery (for roi_periphery method)
            elif bg_method == 'roi_periphery' and n_rois > 0:
                # Create periphery mask for the first selected ROI
                first_roi_idx = roi_indices[0]
                mask = roi_masks[first_roi_idx]
                expanded_mask = binary_dilation(mask, iterations=periphery_size)
                periphery_mask = expanded_mask & ~mask
                
                # Create a visualization of the ROI periphery
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Get first frame for background display
                first_frame = image_data[0]
                
                # Display original ROI
                axes[0].imshow(normalize_for_display(first_frame), cmap='gray')
                axes[0].imshow(mask, cmap='hot', alpha=0.5)
                axes[0].set_title(f'ROI {first_roi_idx+1} Mask')
                axes[0].axis('off')
                
                # Display expanded ROI
                axes[1].imshow(normalize_for_display(first_frame), cmap='gray')
                axes[1].imshow(expanded_mask, cmap='hot', alpha=0.5)
                axes[1].set_title(f'Expanded Mask (periphery={periphery_size})')
                axes[1].axis('off')
                
                # Display periphery only
                axes[2].imshow(normalize_for_display(first_frame), cmap='gray')
                axes[2].imshow(periphery_mask, cmap='hot', alpha=0.5)
                axes[2].set_title('Periphery Mask (for background)')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
            
            # Update config with current values
            if 'background' not in config['roi_processing']:
                config['roi_processing']['background'] = {}
            
            config['roi_processing']['background']['method'] = bg_method
            config['roi_processing']['background']['percentile'] = percentile
            config['roi_processing']['background']['min_background_area'] = min_bg_area
            config['roi_processing']['background']['median_filter_size'] = median_filter_size
            config['roi_processing']['background']['periphery_size'] = periphery_size
            
            print(f"Updated config with: method={bg_method}, percentile={percentile}")
            print(f"min_background_area={min_bg_area}, median_filter_size={median_filter_size}")
            print(f"periphery_size={periphery_size}")
            print("To apply these settings to your pipeline, update your config.yaml file.")
        
        # Create and return interactive widget
        return interactive(
            display_background_subtraction,
            bg_method=bg_method_selector,
            percentile=percentile_slider,
            min_bg_area=min_bg_area_slider,
            median_filter_size=median_filter_slider,
            periphery_size=periphery_size_slider,
            roi_indices=roi_selector
        )
        
    except Exception as e:
        print(f"Error creating background subtraction visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_event_detection_visualization(roi_data, corrected_data, roi_masks, logger, config):
    """Create interactive visualization for event detection and analysis"""
    try:
        # Import needed modules here for self-containment
        from modules.analysis import extract_peak_parameters, extract_spontaneous_activity
        
        # Create a copy of traces for visualization
        # We'll convert ROI data to dF/F for the event detection
        if corrected_data is None:
            # If corrected_data isn't available, use roi_data directly
            traces_for_analysis = roi_data.copy()
        else:
            # Extract traces directly from corrected_data using ROI masks
            n_rois = len(roi_masks)
            n_frames = corrected_data.shape[0]
            traces_for_analysis = np.zeros((n_rois, n_frames), dtype=np.float32)
            for i, mask in enumerate(roi_masks):
                for t in range(n_frames):
                    binary_mask = mask.astype(bool)
                    traces_for_analysis[i, t] = np.mean(corrected_data[t][binary_mask])
        
        # Convert to dF/F using a simple baseline calculation
        # This is just for visualization - the real pipeline will use more sophisticated methods
        df_f_traces = np.zeros_like(traces_for_analysis)
        for i in range(len(traces_for_analysis)):
            # Use first 100 frames or fewer for baseline calculation
            baseline_frames = min(100, traces_for_analysis.shape[1])
            baseline = np.percentile(traces_for_analysis[i, :baseline_frames], 8)
            df_f_traces[i] = (traces_for_analysis[i] - baseline) / baseline if baseline > 0 else traces_for_analysis[i]
        
        # Create widgets for event detection parameters
        # Peak detection parameters
        prominence_slider = widgets.FloatSlider(
            value=0.03,
            min=0.01,
            max=0.2,
            step=0.01,
            description='Prominence:',
            style={'description_width': 'initial'}
        )
        
        width_slider = widgets.IntSlider(
            value=2,
            min=1,
            max=10,
            step=1,
            description='Width:',
            style={'description_width': 'initial'}
        )
        
        distance_slider = widgets.IntSlider(
            value=10,
            min=5,
            max=30,
            step=1,
            description='Distance:',
            style={'description_width': 'initial'}
        )
        
        height_slider = widgets.FloatSlider(
            value=0.02,
            min=0.01,
            max=0.2,
            step=0.01,
            description='Height:',
            style={'description_width': 'initial'}
        )
        
        # Activity threshold
        active_threshold_slider = widgets.FloatSlider(
            value=0.02,
            min=0.01,
            max=0.1,
            step=0.01,
            description='Activity Threshold:',
            style={'description_width': 'initial'}
        )
        
        # Widget for condition selection
        condition_selector = widgets.Dropdown(
            options=[
                ('Spontaneous (0µm)', '0um'),
                ('Evoked (10µm)', '10um'),
                ('Evoked (25µm)', '25um')
            ],
            value='0um',
            description='Condition:',
            style={'description_width': 'initial'}
        )
        
        # Widget to select ROIs to display
        roi_selector = widgets.SelectMultiple(
            options=[(f"ROI {i+1}", i) for i in range(min(10, len(df_f_traces)))],
            value=[0, 1, 2],  # Default: first 3 ROIs
            description='ROIs to Display:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        # Define the display function
        def display_event_detection(prominence, width, distance, height, active_threshold, condition, roi_indices):
            if not roi_indices:
                print("Please select at least one ROI to display")
                return
            
            # Create peak detection config
            peak_config = {
                "prominence": prominence,
                "width": width,
                "distance": distance,
                "height": height,
                "rel_height": 0.5
            }
            
            # Create the peak detection and display
            n_rois = len(roi_indices)
            fig, axes = plt.subplots(n_rois, 1, figsize=(15, 4*n_rois))
            
            # Handle single ROI case
            if n_rois == 1:
                axes = np.array([axes])
            
            # Set analysis frames based on condition
            if condition == '0um':
                # For spontaneous, analyze all frames
                analysis_frames = [0, df_f_traces.shape[1]-1]
                active_metric = "spont_peak_frequency"
                title_suffix = "Spontaneous Activity"
            else:
                # For evoked, focus on frames after stimulus
                analysis_frames = [100, df_f_traces.shape[1]-1]
                active_metric = "peak_amplitude"
                title_suffix = f"Evoked Activity ({condition})"
            
            # Calculate baseline frames - just use first 100 frames or fewer
            baseline_frames = [0, min(100, df_f_traces.shape[1]-1)]
            
            # Process and display each selected ROI
            active_rois = 0
            for i, roi_idx in enumerate(roi_indices):
                trace = df_f_traces[roi_idx]
                
                # Extract analysis window
                analysis_start, analysis_end = analysis_frames
                analysis_window = trace[analysis_start:analysis_end+1]
                
                # For evoked conditions, calculate and display stimulus time
                if condition != '0um':
                    stim_frame = 100  # Frame where stimulus occurs
                
                # Extract peaks
                if condition == '0um':
                    # For spontaneous, look at peaks during baseline period
                    spont_params = extract_spontaneous_activity(
                        trace[baseline_frames[0]:baseline_frames[1]+1],
                        {"prominence": prominence/2, "width": width},  # Use lower threshold for spontaneous
                        logger
                    )
                    is_active = spont_params['peak_frequency'] > active_threshold
                    if is_active:
                        active_rois += 1
                    
                    # Plot trace
                    axes[i].plot(trace, 'k-', label='dF/F')
                    
                    # Find and highlight peaks
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(
                        trace,
                        prominence=prominence/2,
                        width=width,
                        distance=distance,
                        height=active_threshold
                    )
                    
                    if len(peaks) > 0:
                        axes[i].plot(peaks, trace[peaks], 'ro', label='Peaks')
                    
                    # Add title with metrics
                    peak_freq = spont_params['peak_frequency']
                    axes[i].set_title(f"ROI {roi_idx+1} - {'Active' if is_active else 'Inactive'} - Peak Freq: {peak_freq:.2f}/100 frames")
                    
                else:
                    # For evoked, look at peaks after stimulus
                    peak_params = extract_peak_parameters(
                        analysis_window,
                        peak_config,
                        logger
                    )
                    
                    # Check if ROI is active based on peak amplitude
                    is_active = peak_params['amplitude'] > active_threshold
                    if is_active:
                        active_rois += 1
                    
                    # Plot trace
                    axes[i].plot(trace, 'k-', label='dF/F')
                    
                    # Add a vertical line at stimulus time
                    axes[i].axvline(x=stim_frame, color='r', linestyle='--', label='Stimulus', alpha=0.7)
                    
                    # Highlight analysis window
                    axes[i].axvspan(analysis_start, analysis_end, color='lightgray', alpha=0.2, label='Analysis Window')
                    
                    # Find and highlight peaks
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(
                        analysis_window,
                        prominence=prominence,
                        width=width,
                        distance=distance,
                        height=height
                    )
                    
                    if len(peaks) > 0:
                        # Adjust peak indices to match original trace
                        adjusted_peaks = peaks + analysis_start
                        axes[i].plot(adjusted_peaks, trace[adjusted_peaks], 'ro', label='Peaks')
                    
                    # Add title with metrics
                    amplitude = peak_params['amplitude']
                    axes[i].set_title(f"ROI {roi_idx+1} - {'Active' if is_active else 'Inactive'} - Peak Amplitude: {amplitude:.4f}")
                
                # Add zero line for reference
                axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # Add a threshold line
                axes[i].axhline(y=active_threshold, color='g', linestyle=':', 
                               label=f'Threshold ({active_threshold:.2f})', alpha=0.5)
                
                axes[i].set_xlabel('Frame')
                axes[i].set_ylabel('dF/F')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.suptitle(f"Event Detection - {title_suffix} ({active_rois}/{n_rois} ROIs Active)", fontsize=16, y=1.02)
            plt.show()
            
            # Update config with current values
            # Peak detection parameters
            if 'peak_detection' not in config['analysis']:
                config['analysis']['peak_detection'] = {}
            
            config['analysis']['peak_detection']['prominence'] = prominence
            config['analysis']['peak_detection']['width'] = width
            config['analysis']['peak_detection']['distance'] = distance
            config['analysis']['peak_detection']['height'] = height
            
            # Activity threshold
            config['analysis']['active_threshold'] = active_threshold
            
            # Condition-specific parameters
            if 'condition_specific' not in config['analysis']:
                config['analysis']['condition_specific'] = {}
            
            if condition not in config['analysis']['condition_specific']:
                config['analysis']['condition_specific'][condition] = {}
            
            config['analysis']['condition_specific'][condition]['active_threshold'] = active_threshold
            config['analysis']['condition_specific'][condition]['active_metric'] = active_metric
            
            print(f"Updated config with: prominence={prominence}, width={width}, distance={distance}, height={height}")
            print(f"active_threshold={active_threshold}, condition={condition}, active_metric={active_metric}")
            print("To apply these settings to your pipeline, update your config.yaml file.")
        
        # Create and return interactive widget
        return interactive(
            display_event_detection,
            prominence=prominence_slider,
            width=width_slider,
            distance=distance_slider,
            height=height_slider,
            active_threshold=active_threshold_slider,
            condition=condition_selector,
            roi_indices=roi_selector
        )
        
    except Exception as e:
        print(f"Error creating event detection visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None