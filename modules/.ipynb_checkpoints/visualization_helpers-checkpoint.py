#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Helpers Module
--------------------------
Interactive visualizations for the fluorescence pipeline using HoloViews and Panel.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import find_peaks

# Import necessary visualization libraries
try:
    import holoviews as hv
    import panel as pn
    import param
    HAS_PANEL = True
except ImportError:
    HAS_PANEL = False
    print("HoloViews and Panel are required for interactive visualization.")
    print("Install with: pip install holoviews panel param")

def normalize_for_display(img):
    """Normalize image for display"""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def run_denoising_visualization(image_data, config, logger=None):
    """
    Create interactive visualization for denoising methods.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    panel.viewable.Viewable or None
        Panel app for visualization
    """
    if not HAS_PANEL:
        if logger:
            logger.warning("Panel and HoloViews are required for this visualization.")
        return None
    
    if image_data is None:
        if logger:
            logger.warning("No image data provided.")
        return None
    
    # Initialize holoviews with bokeh backend
    hv.extension('bokeh')
    
    try:
        # Create parameter widgets
        frame_slider = pn.widgets.IntSlider(
            name='Frame', 
            start=0, 
            end=min(image_data.shape[0]-1, 100), 
            step=1, 
            value=10
        )
        
        # Method selector
        method_selector = pn.widgets.Select(
            name='Method',
            options={
                'Gaussian': 'gaussian',
                'Median': 'median',
                'Bilateral': 'bilateral',
                'Anisotropic': 'anisotropic'
            },
            value='gaussian'
        )
        
        # Common parameters
        ksize_slider = pn.widgets.IntSlider(
            name='Kernel Size',
            start=1,
            end=21,
            step=2,
            value=5
        )
        
        # Gaussian parameters
        sigmaX_slider = pn.widgets.FloatSlider(
            name='Sigma X',
            start=0.1,
            end=10.0,
            step=0.1,
            value=1.5
        )
        
        # Bilateral parameters
        sigmaColor_slider = pn.widgets.FloatSlider(
            name='Sigma Color',
            start=10,
            end=150,
            step=5,
            value=75
        )
        
        sigmaSpace_slider = pn.widgets.FloatSlider(
            name='Sigma Space',
            start=10,
            end=150,
            step=5,
            value=75
        )
        
        # Anisotropic parameters
        niter_slider = pn.widgets.IntSlider(
            name='Iterations',
            start=1,
            end=20,
            step=1,
            value=5
        )
        
        kappa_slider = pn.widgets.FloatSlider(
            name='Kappa',
            start=10,
            end=100,
            step=5,
            value=50
        )
        
        # Parameter settings based on method
        method_params = {
            'gaussian': pn.Column(ksize_slider, sigmaX_slider),
            'median': pn.Column(ksize_slider),
            'bilateral': pn.Column(ksize_slider, sigmaColor_slider, sigmaSpace_slider),
            'anisotropic': pn.Column(niter_slider, kappa_slider)
        }
        
        # Container for method-specific parameters
        param_container = pn.Column(method_params['gaussian'])
        
        # Function to update parameter container
        def update_param_container(event):
            param_container.clear()
            param_container.append(method_params[event.new])
        
        method_selector.param.watch(update_param_container, 'value')
        
        # Main function to apply denoising
        @pn.depends(frame_slider, method_selector, ksize_slider, sigmaX_slider, 
                   sigmaColor_slider, sigmaSpace_slider, niter_slider, kappa_slider)
        def apply_denoising(frame_idx, method, ksize, sigmaX, sigmaColor, sigmaSpace, niter, kappa):
            # Ensure ksize is odd for methods that require it
            if method in ['gaussian', 'median'] and ksize % 2 == 0:
                ksize += 1
                
            # Get the selected frame
            sample_frame = image_data[frame_idx].copy()
            
            # Set up parameters based on method
            params = {}
            method_title = ""
            
            if method == 'gaussian':
                params = {'ksize': (ksize, ksize), 'sigmaX': sigmaX}
                method_title = f'Gaussian Denoised (k={ksize}, σ={sigmaX:.1f})'
            elif method == 'median':
                params = {'ksize': ksize}
                method_title = f'Median Denoised (k={ksize})'
            elif method == 'bilateral':
                params = {'d': ksize, 'sigmaColor': sigmaColor, 'sigmaSpace': sigmaSpace}
                method_title = f'Bilateral Denoised (d={ksize}, σC={sigmaColor:.1f}, σS={sigmaSpace:.1f})'
            elif method == 'anisotropic':
                params = {'niter': niter, 'kappa': kappa, 'gamma': 0.1, 'option': 1}
                method_title = f'Anisotropic Denoised (iter={niter}, κ={kappa:.1f})'
            
            # Apply denoising
            import time
            start_time = time.time()
            
            try:
                denoised_frame = denoise_frame(sample_frame, method, **params)
                process_time = time.time() - start_time
                
                # Calculate difference
                diff = np.abs(sample_frame - denoised_frame)
                
                # Normalize for display
                norm_orig = normalize_for_display(sample_frame)
                norm_denoised = normalize_for_display(denoised_frame)
                norm_diff = normalize_for_display(diff)
                
                # Convert to HoloViews Image objects
                orig_img = hv.Image(norm_orig).opts(
                    title='Original Frame',
                    cmap='gray', 
                    width=300, 
                    height=300
                )
                
                denoised_img = hv.Image(norm_denoised).opts(
                    title=f'{method_title}\nProcess time: {process_time:.3f}s',
                    cmap='gray', 
                    width=300, 
                    height=300
                )
                
                diff_img = hv.Image(norm_diff).opts(
                    title='Difference (Red=More Change)',
                    cmap='hot', 
                    width=300, 
                    height=300
                )
                
                # Update config with current values
                if 'denoise' not in config['preprocessing']:
                    config['preprocessing']['denoise'] = {}
                
                config['preprocessing']['denoise']['enabled'] = True
                config['preprocessing']['denoise']['method'] = method
                
                if method == 'gaussian':
                    config['preprocessing']['denoise']['params'] = {
                        'ksize': [ksize, ksize],
                        'sigmaX': sigmaX
                    }
                elif method == 'median':
                    config['preprocessing']['denoise']['params'] = {
                        'ksize': ksize
                    }
                elif method == 'bilateral':
                    config['preprocessing']['denoise']['params'] = {
                        'd': ksize,
                        'sigmaColor': sigmaColor,
                        'sigmaSpace': sigmaSpace
                    }
                elif method == 'anisotropic':
                    config['preprocessing']['denoise']['params'] = {
                        'niter': niter,
                        'kappa': kappa,
                        'gamma': 0.1,
                        'option': 1
                    }
                
                # Return layout with all three images
                return (orig_img + denoised_img + diff_img).cols(3)
                
            except Exception as e:
                error_text = f"Error applying {method} denoising: {str(e)}"
                if logger:
                    logger.error(error_text)
                return hv.Text(0, 0, error_text).opts(width=900, height=300)
        
        # Create method descriptions
        method_descriptions = pn.pane.HTML("""
        <h3>Denoising Methods:</h3>
        <ul>
            <li><strong>Gaussian</strong>: Blurs the image using a Gaussian filter. Good for general noise reduction while preserving edges better than simple averaging.</li>
            <li><strong>Median</strong>: Replaces each pixel with the median of neighboring pixels. Excellent for removing salt-and-pepper noise while preserving edges.</li>
            <li><strong>Bilateral</strong>: Edge-preserving smoothing that reduces noise while maintaining sharp edges by considering both spatial proximity and intensity difference.</li>
            <li><strong>Anisotropic</strong>: Advanced edge-preserving smoothing that adapts to local image structure. Great for preserving fine details while removing noise.</li>
        </ul>
        <p><em>Note: Anisotropic diffusion is computationally intensive and may take longer to process.</em></p>
        """)
        
        # Create a Panel app with the widgets and visualization
        app = pn.Column(
            "## Denoising Tool",
            method_descriptions,
            pn.Row(
                pn.Column(frame_slider, method_selector, param_container, width=300),
                apply_denoising
            )
        )
        
        return app
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating denoising visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        return None

def denoise_frame(frame, method, **kwargs):
    """
    Denoise a single frame using various methods.
    This is a simplified version for visualization.
    """
    if method == "gaussian":
        ksize = kwargs.get('ksize', (5, 5))
        sigma = kwargs.get('sigmaX', 0)
        return cv2.GaussianBlur(frame, ksize, sigma)
    
    elif method == "median":
        ksize = kwargs.get('ksize', 5)
        if isinstance(ksize, int) and ksize % 2 == 0:
            ksize += 1  # Make sure it's odd
        return cv2.medianBlur(frame, ksize)
    
    elif method == "bilateral":
        d = kwargs.get('d', 9)
        sigmaColor = kwargs.get('sigmaColor', 75)
        sigmaSpace = kwargs.get('sigmaSpace', 75)
        return cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
    
    elif method == "anisotropic":
        # Import anisotropic_diffusion function
        try:
            from medpy.filter.smoothing import anisotropic_diffusion
        except ImportError:
            raise ImportError("medpy is required for anisotropic diffusion. Install with: pip install medpy")
        
        niter = kwargs.get('niter', 5)
        kappa = kwargs.get('kappa', 50)
        gamma = kwargs.get('gamma', 0.1)
        option = kwargs.get('option', 1)
        
        # Convert frame to float for anisotropic diffusion if needed
        if frame.dtype != np.float32 and frame.dtype != np.float64:
            frame = frame.astype(np.float32)
            
        # Call anisotropic_diffusion with correct parameter types
        return anisotropic_diffusion(frame, niter=niter, kappa=kappa, gamma=gamma, option=option)
    
    else:
        raise ValueError(f"Denoising method '{method}' not understood")

def run_background_removal_visualization(image_data, config, logger=None):
    """
    Create interactive visualization for background removal.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    panel.viewable.Viewable or None
        Panel app for visualization
    """
    if not HAS_PANEL:
        if logger:
            logger.warning("Panel and HoloViews are required for this visualization.")
        return None
    
    if image_data is None:
        if logger:
            logger.warning("No image data provided.")
        return None
    
    # Initialize holoviews with bokeh backend
    hv.extension('bokeh')
    
    try:
        # Create parameter widgets
        frame_slider = pn.widgets.IntSlider(
            name='Frame', 
            start=0, 
            end=min(image_data.shape[0]-1, 100), 
            step=1, 
            value=10
        )
        
        # Method selector
        method_selector = pn.widgets.Select(
            name='Method',
            options={
                'Uniform': 'uniform',
                'Tophat': 'tophat'
            },
            value='uniform'
        )
        
        # Window size parameter
        window_size_slider = pn.widgets.IntSlider(
            name='Window Size',
            start=3,
            end=21,
            step=2,
            value=7
        )
        
        # Main function to apply background removal
        @pn.depends(frame_slider, method_selector, window_size_slider)
        def apply_background_removal(frame_idx, method, window_size):
            # Get the selected frame
            sample_frame = image_data[frame_idx].copy()
            
            # For tophat, create disk element
            selem = None
            if method == 'tophat':
                # Use skimage's disk function if available
                try:
                    from skimage.morphology import disk
                    selem = disk(window_size)
                except ImportError:
                    if logger:
                        logger.warning("skimage.morphology.disk not available. Using default structuring element.")
            
            # Apply background removal
            import time
            start_time = time.time()
            
            try:
                bg_removed_frame = remove_background_perframe(sample_frame, method, window_size, selem)
                process_time = time.time() - start_time
                
                # Calculate difference
                diff = np.abs(sample_frame - bg_removed_frame)
                
                # Get a central row for profile
                center_row = sample_frame.shape[0] // 2
                original_profile = sample_frame[center_row, :]
                bg_removed_profile = bg_removed_frame[center_row, :]
                
                # Normalize for display
                norm_orig = normalize_for_display(sample_frame)
                norm_bg_removed = normalize_for_display(bg_removed_frame)
                norm_diff = normalize_for_display(diff)
                
                # Convert to HoloViews objects
                orig_img = hv.Image(norm_orig).opts(
                    title='Original Frame',
                    cmap='gray', 
                    width=300, 
                    height=300
                )
                
                bg_removed_img = hv.Image(norm_bg_removed).opts(
                    title=f'{method.title()} Background Removal\nWindow size: {window_size}, Time: {process_time:.3f}s',
                    cmap='gray', 
                    width=300, 
                    height=300
                )
                
                diff_img = hv.Image(norm_diff).opts(
                    title='Difference (Red=Removed Background)',
                    cmap='hot', 
                    width=300, 
                    height=300
                )
                
                # Create profile comparison
                x_coords = np.arange(len(original_profile))
                profile_orig = hv.Curve((x_coords, original_profile), 'Pixel', 'Intensity').opts(
                    color='blue', line_width=2, alpha=0.8, label='Original'
                )
                profile_bg = hv.Curve((x_coords, bg_removed_profile), 'Pixel', 'Intensity').opts(
                    color='red', line_width=2, alpha=0.8, label='Background Removed'
                )
                
                profile_plot = (profile_orig * profile_bg).opts(
                    title='Intensity Profile (Center Row)',
                    width=600,
                    height=300,
                    show_grid=True,
                    legend_position='top_right'
                )
                
                # Update config with current values
                if 'background_removal' not in config['preprocessing']:
                    config['preprocessing']['background_removal'] = {}
                
                config['preprocessing']['background_removal']['enabled'] = True
                config['preprocessing']['background_removal']['method'] = method
                config['preprocessing']['background_removal']['window_size'] = window_size
                
                # Return layout with images and profile
                images = (orig_img + bg_removed_img + diff_img).cols(3)
                return pn.Column(images, profile_plot)
                
            except Exception as e:
                error_text = f"Error applying {method} background removal: {str(e)}"
                if logger:
                    logger.error(error_text)
                return hv.Text(0, 0, error_text).opts(width=900, height=300)
        
        # Create method descriptions
        method_descriptions = pn.pane.HTML("""
        <h3>Background Removal Methods:</h3>
        <ul>
            <li><strong>Uniform</strong>: Removes background by subtracting a uniformly blurred version of the image. Good for removing slow variations in background intensity.</li>
            <li><strong>Tophat</strong>: Uses morphological top-hat transformation to enhance small features while removing background. Excellent for extracting small bright features from varying backgrounds.</li>
        </ul>
        <p><em>Adjust the window size parameter to control how much background is removed. Larger values remove more broadly distributed background structures.</em></p>
        """)
        
        # Create a Panel app with the widgets and visualization
        app = pn.Column(
            "## Background Removal Tool",
            method_descriptions,
            pn.Row(
                pn.Column(frame_slider, method_selector, window_size_slider, width=300),
                apply_background_removal
            )
        )
        
        return app
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating background removal visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        return None

def remove_background_perframe(fm, method, wnd=None, selem=None):
    """
    Remove background from a single frame.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.
    method : str
        Method to use to remove background. Should be either "uniform" or "tophat".
    wnd : int, optional
        Size of the uniform filter. Only used if method == "uniform".
    selem : np.ndarray, optional
        Kernel used for morphological operations. Only used if method == "tophat".

    Returns
    -------
    fm : np.ndarray
        The frame with background removed.
    """
    if method == "uniform":
        if wnd is None:
            raise ValueError("Window size (wnd) must be provided for uniform method")
        from scipy.ndimage import uniform_filter
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        if selem is None:
            # Create a default disk element if not provided
            try:
                from skimage.morphology import disk
                selem = disk(3)  # Default radius of 3
            except ImportError:
                # If skimage is not available, create a simple circular mask
                radius = 3
                y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                selem = (x**2 + y**2 <= radius**2).astype(np.uint8)
        
        return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)
    else:
        raise ValueError(f"Background removal method '{method}' not understood")

def run_motion_correction_visualization(image_data, tif_path, config, logger=None):
    """
    Create interactive visualization for motion correction.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    tif_path : str
        Path to the original TIFF file
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    panel.viewable.Viewable or None
        Panel app for visualization
    """
    if not HAS_PANEL:
        if logger:
            logger.warning("Panel and HoloViews are required for this visualization.")
        return None
    
    if image_data is None:
        if logger:
            logger.warning("No image data provided.")
        return None
    
    # Initialize holoviews with bokeh backend
    hv.extension('bokeh')
    
    try:
        # Check if NoRMCorre is available
        try:
            from modules.motion_correction import apply_normcorre_correction, estimate_motion
            has_normcorre = True
        except ImportError:
            has_normcorre = False
            warning_text = "Warning: NoRMCorre module not found. Run without correction to see visualization only."
            if logger:
                logger.warning(warning_text)
        
        # Create parameter widgets
        method_selector = pn.widgets.Select(
            name='Method',
            options={
                'Rigid': 'rigid',
                'Non-rigid': 'nonrigid'
            },
            value='rigid'
        )
        
        max_shift_slider = pn.widgets.IntSlider(
            name='Max Shift (pixels)',
            start=5,
            end=30,
            step=1,
            value=10
        )
        
        patch_size_slider = pn.widgets.IntSlider(
            name='Patch Size',
            start=32,
            end=200,
            step=8,
            value=50
        )
        
        # Button to run motion correction
        run_button = pn.widgets.Button(
            name='Run Motion Correction',
            button_type='primary',
            width=200
        )
        
        # Create results container
        results_container = pn.Column()
        
        # Variables to store correction results
        correction_results = {'corrected_data': None, 'shifts': None, 'is_corrected': False}
        
        # Function to run motion correction
        def run_correction(event):
            results_container.clear()
            results_container.append(pn.pane.Markdown("## Running motion correction..."))
            
            if not has_normcorre:
                results_container.clear()
                results_container.append(pn.pane.Markdown(
                    "## Motion correction module not available\n\nPlease ensure NoRMCorre is installed."
                ))
                return
            
            try:
                # Setup parameters
                motion_params = {
                    'max_shift': max_shift_slider.value,
                    'patch_size': [patch_size_slider.value, patch_size_slider.value],
                    'method': method_selector.value
                }
                
                message = pn.pane.Markdown(f"""
                ## Running motion correction...
                Method: {method_selector.value}
                Max shift: {max_shift_slider.value}
                Patch size: {patch_size_slider.value}
                
                This may take several minutes for non-rigid correction on large datasets.
                """)
                
                results_container.clear()
                results_container.append(message)
                
                # Run motion correction asynchronously
                import threading
                
                def process_correction():
                    try:
                        # Run motion correction
                        import time
                        start_time = time.time()
                        
                        # Apply motion correction using the module
                        corrected_data, shifts = apply_normcorre_correction(
                            image_data, 
                            motion_params, 
                            logger
                        )
                        
                        processing_time = time.time() - start_time
                        
                        # Store results
                        correction_results['corrected_data'] = corrected_data
                        correction_results['shifts'] = shifts
                        correction_results['is_corrected'] = True
                        
                        # Update config
                        if 'motion_correction' not in config:
                            config['motion_correction'] = {}
                        
                        config['motion_correction']['enabled'] = True
                        config['motion_correction']['method'] = method_selector.value
                        config['motion_correction']['max_shift'] = max_shift_slider.value
                        config['motion_correction']['patch_size'] = patch_size_slider.value
                        
                        # Update results container with visualizations
                        results_container.clear()
                        results_container.append(
                            pn.pane.Markdown(f"## Motion Correction Completed\nTime: {processing_time:.2f} seconds")
                        )
                        
                        # Create visualization
                        plot = create_motion_correction_plot(
                            image_data, 
                            corrected_data, 
                            shifts, 
                            method_selector.value
                        )
                        results_container.append(plot)
                        
                        # Add frame comparison slider
                        frame_results = create_frame_comparison(image_data, corrected_data, shifts, method_selector.value)
                        results_container.append(frame_results)
                        
                        # Add save button
                        save_button = pn.widgets.Button(
                            name='Save Corrected Data',
                            button_type='success',
                            width=200
                        )
                        
                        def on_save(event):
                            import os
                            from pathlib import Path
                            import tifffile
                            
                            # Generate output path
                            output_path = os.path.join(os.path.dirname(tif_path), 
                                                    f"{Path(tif_path).stem}_mc.tif")
                            
                            save_status = pn.pane.Markdown(f"Saving to {output_path}...")
                            results_container.append(save_status)
                            
                            try:
                                # Save as TIFF
                                tifffile.imwrite(output_path, corrected_data)
                                save_status.object = f"Saved successfully to {output_path}"
                            except Exception as e:
                                save_status.object = f"Error saving file: {str(e)}"
                        
                        save_button.on_click(on_save)
                        results_container.append(save_button)
                        
                    except Exception as e:
                        error_message = f"Error during motion correction: {str(e)}"
                        if logger:
                            logger.error(error_message)
                        results_container.clear()
                        results_container.append(pn.pane.Markdown(f"## Error\n{error_message}"))
                
                # Start processing in a thread to keep UI responsive
                thread = threading.Thread(target=process_correction)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                error_message = f"Error starting motion correction: {str(e)}"
                if logger:
                    logger.error(error_message)
                results_container.clear()
                results_container.append(pn.pane.Markdown(f"## Error\n{error_message}"))
        
        # Attach button click handler
        run_button.on_click(run_correction)
        
        # Create method descriptions
        method_descriptions = pn.pane.HTML("""
        <h3>Motion Correction using NoRMCorre:</h3>
        <p>NoRMCorre is a non-rigid motion correction algorithm for calcium imaging data. It can correct for both rigid and non-rigid motion.</p>
        <ul>
            <li><strong>Rigid</strong>: Corrects for whole-frame shifts in X and Y. Faster but less accurate for complex motion.</li>
            <li><strong>Non-rigid</strong>: Divides the image into patches and corrects each patch separately. Better for complex distortions but slower.</li>
        </ul>
        <p><strong>Parameters:</strong></p>
        <ul>
            <li><strong>Max Shift</strong>: Maximum allowed shift in pixels. Limits how far the algorithm can move pixels.</li>
            <li><strong>Patch Size</strong>: Size of patches for non-rigid correction (in pixels). Smaller patches can correct more local motion but may introduce artifacts.</li>
        </ul>
        <p><em>Note: Non-rigid correction can take several minutes to run depending on dataset size.</em></p>
        """)
        
        # Create a Panel app with the widgets and visualization
        app = pn.Column(
            "## Motion Correction Tool",
            method_descriptions,
            pn.Row(
                pn.Column(method_selector, max_shift_slider, patch_size_slider, run_button, width=300),
                results_container
            )
        )
        
        return app
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating motion correction visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        return None

def create_motion_correction_plot(original_data, corrected_data, shifts, method):
    """
    Create plots for motion correction results.
    
    Parameters
    ----------
    original_data : numpy.ndarray
        Original image data
    corrected_data : numpy.ndarray
        Motion-corrected image data
    shifts : numpy.ndarray
        Motion shifts
    method : str
        Motion correction method ('rigid' or 'nonrigid')
        
    Returns
    -------
    panel.viewable.Viewable
        Panel layout with plots
    """
    # Create shift plots
    if method == 'rigid':
        # Plot X and Y shifts over time for rigid correction
        x_shifts = hv.Curve((np.arange(len(shifts)), shifts[:, 0]), 'Frame', 'Shift (pixels)').opts(
            color='red', line_width=2, alpha=0.8, label='X shifts'
        )
        y_shifts = hv.Curve((np.arange(len(shifts)), shifts[:, 1]), 'Frame', 'Shift (pixels)').opts(
            color='blue', line_width=2, alpha=0.8, label='Y shifts'
        )
        
        shift_plot = (x_shifts * y_shifts).opts(
            title='Rigid Motion Shifts',
            width=400,
            height=300,
            show_grid=True,
            legend_position='top_right'
        )
        
        # Calculate magnitude of shifts
        shift_magnitude = np.sqrt(shifts[:, 0]**2 + shifts[:, 1]**2)
        magnitude_plot = hv.Curve((np.arange(len(shifts)), shift_magnitude), 'Frame', 'Magnitude (pixels)').opts(
            color='green',
            line_width=2,
            title='Motion Magnitude',
            width=400,
            height=300,
            show_grid=True
        )
    else:
        # For non-rigid, plot average shifts
        avg_shifts = np.mean(shifts, axis=(1, 2))
        x_shifts = hv.Curve((np.arange(len(avg_shifts)), avg_shifts[:, 0]), 'Frame', 'Shift (pixels)').opts(
            color='red', line_width=2, alpha=0.8, label='Avg X shifts'
        )
        y_shifts = hv.Curve((np.arange(len(avg_shifts)), avg_shifts[:, 1]), 'Frame', 'Shift (pixels)').opts(
            color='blue', line_width=2, alpha=0.8, label='Avg Y shifts'
        )
        
        shift_plot = (x_shifts * y_shifts).opts(
            title='Non-Rigid Average Motion Shifts',
            width=400,
            height=300,
            show_grid=True,
            legend_position='top_right'
        )
        
        # Calculate magnitude of average shifts
        shift_magnitude = np.sqrt(avg_shifts[:, 0]**2 + avg_shifts[:, 1]**2)
        magnitude_plot = hv.Curve((np.arange(len(shift_magnitude)), shift_magnitude), 'Frame', 'Magnitude (pixels)').opts(
            color='green',
            line_width=2,
            title='Average Motion Magnitude',
            width=400,
            height=300,
            show_grid=True
        )
    
    # Before/After comparison - show max projection
    orig_max = np.max(original_data, axis=0)
    corr_max = np.max(corrected_data, axis=0)
    
    # Normalize for display
    norm_orig_max = normalize_for_display(orig_max)
    norm_corr_max = normalize_for_display(corr_max)
    
    # Create image objects
    orig_img = hv.Image(norm_orig_max).opts(
        title='Maximum Projection (Before Correction)',
        cmap='gray',
        width=400,
        height=400
    )
    
    corr_img = hv.Image(norm_corr_max).opts(
        title='Maximum Projection (After Correction)',
        cmap='gray',
        width=400,
        height=400
    )
    
    # Combine plots
    top_row = pn.Row(shift_plot, magnitude_plot)
    bottom_row = pn.Row(orig_img, corr_img)
    
    return pn.Column(top_row, bottom_row)

def create_frame_comparison(original_data, corrected_data, shifts, method):
    """
    Create interactive frame comparison for motion correction.
    
    Parameters
    ----------
    original_data : numpy.ndarray
        Original image data
    corrected_data : numpy.ndarray
        Motion-corrected image data
    shifts : numpy.ndarray
        Motion shifts
    method : str
        Motion correction method ('rigid' or 'nonrigid')
        
    Returns
    -------
    panel.viewable.Viewable
        Panel layout with interactive frame comparison
    """
    frame_slider = pn.widgets.IntSlider(
        name='Frame',
        start=0,
        end=len(original_data)-1,
        step=1,
        value=0
    )
    
    @pn.depends(frame_slider)
    def compare_frames(frame_idx):
        # Get original and corrected frames
        original_frame = original_data[frame_idx]
        corrected_frame = corrected_data[frame_idx]
        
        # Normalize for display
        norm_orig = normalize_for_display(original_frame)
        norm_corr = normalize_for_display(corrected_frame)
        
        # Create image objects
        orig_img = hv.Image(norm_orig).opts(
            title=f'Original (Frame {frame_idx})',
            cmap='gray',
            width=400,
            height=400
        )
        
        corr_img = hv.Image(norm_corr).opts(
            title=f'Motion Corrected (Frame {frame_idx})',
            cmap='gray',
            width=400,
            height=400
        )
        
        # Get shift information for this frame
        if method == 'rigid':
            shift_info = f"Frame {frame_idx} shifts: X={shifts[frame_idx, 0]:.2f}, Y={shifts[frame_idx, 1]:.2f}"
        else:
            avg_x = np.mean(shifts[frame_idx, :, :, 0])
            avg_y = np.mean(shifts[frame_idx, :, :, 1])
            shift_info = f"Frame {frame_idx} average shifts: X={avg_x:.2f}, Y={avg_y:.2f}"
        
        # Combine images and shift info
        return pn.Column(
            pn.Row(orig_img, corr_img),
            pn.pane.Markdown(shift_info)
        )
    
    # Create title and header
    header = pn.pane.Markdown("## Frame-by-frame Comparison")
    
    # Combine everything
    return pn.Column(header, frame_slider, compare_frames)

def run_event_detection_visualization(traces, roi_masks, config, logger=None):
    """
    Create interactive visualization for event detection and analysis.
    
    Parameters
    ----------
    traces : numpy.ndarray
        Fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger object
        
    Returns
    -------
    panel.viewable.Viewable or None
        Panel app for visualization
    """
    if not HAS_PANEL:
        if logger:
            logger.warning("Panel and HoloViews are required for this visualization.")
        return None
    
    if traces is None:
        if logger:
            logger.warning("No trace data provided.")
        return None
    
    # Initialize holoviews with bokeh backend
    hv.extension('bokeh')
    
    try:
        # Import needed functions from analysis module
        from modules.analysis import extract_peak_parameters, extract_spontaneous_activity
        
        # Create a copy of traces for visualization
        # Convert to dF/F for visualization if not already
        df_f_traces = traces.copy()
        
        # Create parameter widgets
        # Peak detection parameters
        prominence_slider = pn.widgets.FloatSlider(
            name='Prominence',
            start=0.01,
            end=0.2,
            step=0.01,
            value=0.03
        )
        
        width_slider = pn.widgets.IntSlider(
            name='Width',
            start=1,
            end=10,
            step=1,
            value=2
        )
        
        distance_slider = pn.widgets.IntSlider(
            name='Distance',
            start=5,
            end=30,
            step=1,
            value=10
        )
        
        height_slider = pn.widgets.FloatSlider(
            name='Height',
            start=0.01,
            end=0.2,
            step=0.01,
            value=0.02
        )
        
        # Activity threshold
        active_threshold_slider = pn.widgets.FloatSlider(
            name='Activity Threshold',
            start=0.01,
            end=0.1,
            step=0.01,
            value=0.02
        )
        
        # Condition selector
        condition_selector = pn.widgets.Select(
            name='Condition',
            options={
                'Spontaneous (0µm)': '0um',
                'Evoked (10µm)': '10um',
                'Evoked (25µm)': '25um'
            },
            value='0um'
        )
        
        # ROI selector
        roi_options = {f"ROI {i+1}": i for i in range(min(10, len(df_f_traces)))}
        roi_selector = pn.widgets.MultiSelect(
            name='ROIs to Display',
            options=roi_options,
            value=[0, 1, 2],
            size=5
        )
        
        # Main function to detect events and visualize
        @pn.depends(prominence_slider, width_slider, distance_slider, height_slider, 
                   active_threshold_slider, condition_selector, roi_selector)
        def detect_events(prominence, width, distance, height, active_threshold, condition, roi_indices):
            if not roi_indices:
                return pn.pane.Markdown("Please select at least one ROI to display")
            
            # Create peak detection config
            peak_config = {
                "prominence": prominence,
                "width": width,
                "distance": distance,
                "height": height,
                "rel_height": 0.5
            }
            
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
            
            # Create plots for each selected ROI
            plots = []
            active_rois = 0
            
            for roi_idx in roi_indices:
                trace = df_f_traces[roi_idx]
                
                # Extract analysis window
                analysis_start, analysis_end = analysis_frames
                analysis_window = trace[analysis_start:analysis_end+1]
                
                # Create time values (frames)
                x_coords = np.arange(len(trace))
                
                # Initialize curves
                trace_curve = hv.Curve((x_coords, trace), 'Frame', 'dF/F').opts(
                    color='black', line_width=2, alpha=0.8, label='dF/F'
                )
                
                # Add stimulus line for evoked conditions
                if condition != '0um':
                    stim_frame = 100  # Frame where stimulus occurs
                    stim_line = hv.VLine(stim_frame).opts(
                        color='red', line_width=2, line_dash='dashed', alpha=0.7
                    )
                    
                    # Highlight analysis window
                    analysis_highlight = hv.VSpan(analysis_start, analysis_end).opts(
                        color='lightgray', alpha=0.2
                    )
                    
                    base_plot = trace_curve * stim_line * analysis_highlight
                else:
                    base_plot = trace_curve
                
                # Add threshold line
                threshold_line = hv.HLine(active_threshold).opts(
                    color='green', line_width=1.5, line_dash='dotted', alpha=0.7
                )
                base_plot = base_plot * threshold_line
                
                # Detect peaks for visualization
                from scipy.signal import find_peaks
                
                if condition == '0um':
                    # For spontaneous, check peaks during baseline period
                    spont_params = extract_spontaneous_activity(
                        trace[baseline_frames[0]:baseline_frames[1]+1],
                        {"prominence": prominence/2, "width": width},  # Use lower threshold for spontaneous
                        logger
                    )
                    is_active = spont_params['peak_frequency'] > active_threshold
                    if is_active:
                        active_rois += 1
                    
                    # Find peaks with lower threshold for spontaneous
                    peaks, _ = find_peaks(
                        trace,
                        prominence=prominence/2,
                        width=width,
                        distance=distance,
                        height=active_threshold
                    )
                    
                    # Add detected peaks
                    if len(peaks) > 0:
                        peak_points = hv.Points((peaks, trace[peaks]), 'Frame', 'dF/F').opts(
                            color='red', size=8, alpha=0.8, marker='o'
                        )
                        plot = base_plot * peak_points
                    else:
                        plot = base_plot
                    
                    # Set plot title
                    peak_freq = spont_params['peak_frequency']
                    title = f"ROI {roi_idx+1} - {'Active' if is_active else 'Inactive'} - Peak Freq: {peak_freq:.2f}/100 frames"
                    
                else:
                    # For evoked, detect peaks after stimulus
                    peak_params = extract_peak_parameters(
                        analysis_window,
                        peak_config,
                        logger
                    )
                    
                    # Check if ROI is active based on peak amplitude
                    is_active = peak_params['amplitude'] > active_threshold
                    if is_active:
                        active_rois += 1
                    
                    # Find peaks in the analysis window
                    peaks, _ = find_peaks(
                        analysis_window,
                        prominence=prominence,
                        width=width,
                        distance=distance,
                        height=height
                    )
                    
                    # Add detected peaks to plot
                    if len(peaks) > 0:
                        # Adjust peak indices to match original trace
                        adjusted_peaks = peaks + analysis_start
                        peak_points = hv.Points((adjusted_peaks, trace[adjusted_peaks]), 'Frame', 'dF/F').opts(
                            color='red', size=8, alpha=0.8, marker='o'
                        )
                        plot = base_plot * peak_points
                    else:
                        plot = base_plot
                    
                    # Set plot title
                    amplitude = peak_params['amplitude']
                    title = f"ROI {roi_idx+1} - {'Active' if is_active else 'Inactive'} - Peak Amplitude: {amplitude:.4f}"
                
                # Customize plot options
                plot = plot.opts(
                    title=title,
                    width=800,
                    height=200,
                    show_grid=True,
                    legend_position='top_right'
                )
                
                plots.append(plot)
            
            # Create layout with all plots
            layout = pn.Column(*plots)
            
            # Add summary header
            header = pn.pane.Markdown(f"## Event Detection - {title_suffix}\n{active_rois}/{len(roi_indices)} ROIs Active")
            
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
            
            return pn.Column(header, layout)
        
        # Create method descriptions
        method_descriptions = pn.pane.HTML("""
        <h3>Event Detection Parameters:</h3>
        <ul>
            <li><strong>Prominence</strong>: Minimum vertical distance between a peak and its neighboring valleys. Higher values detect only more significant peaks.</li>
            <li><strong>Width</strong>: Minimum width of peaks in frames. Increase to detect broader peaks.</li>
            <li><strong>Distance</strong>: Minimum distance between peaks in frames. Increase to avoid detecting multiple peaks in one event.</li>
            <li><strong>Height</strong>: Minimum height threshold for peaks. Peaks below this value are ignored.</li>
            <li><strong>Activity Threshold</strong>: Threshold to determine if an ROI is considered active.</li>
        </ul>
        <p><em>Select the appropriate condition to adjust analysis parameters. For spontaneous activity (0µm), all frames are analyzed. For evoked activity (10µm, 25µm), frames after stimulus are analyzed.</em></p>
        """)
        
        # Create a Panel app with the widgets and visualization
        app = pn.Column(
            "## Event Detection Tool",
            method_descriptions,
            pn.Row(
                pn.Column(
                    condition_selector,
                    pn.pane.Markdown("### Peak Detection Parameters"),
                    prominence_slider,
                    width_slider,
                    distance_slider,
                    height_slider,
                    pn.pane.Markdown("### Activity Settings"),
                    active_threshold_slider,
                    pn.pane.Markdown("### ROI Selection"),
                    roi_selector,
                    width=300
                ),
                detect_events
            )
        )
        
        return app
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating event detection visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        return None