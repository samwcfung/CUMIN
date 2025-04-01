# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import holoviews as hv
import panel as pn
from IPython.display import display

# Initialize HoloViews with bokeh backend
hv.extension('bokeh')

def normalize_for_display(img):
    """Normalize image for display"""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def denoise_image(frame, method, **params):
    """Apply denoising method to an image frame"""
    if method == 'gaussian':
        ksize = params.get('ksize', (5, 5))
        sigma = params.get('sigma', 1.5)
        return cv2.GaussianBlur(frame, ksize, sigma)
    
    elif method == 'median':
        ksize = params.get('ksize', 5)
        if ksize % 2 == 0:  # Ensure odd kernel size
            ksize += 1
        return cv2.medianBlur(frame.astype(np.uint8), ksize).astype(frame.dtype)
    
    elif method == 'bilateral':
        d = params.get('d', 9)
        sigmaColor = params.get('sigmaColor', 75)
        sigmaSpace = params.get('sigmaSpace', 75)
        return cv2.bilateralFilter(frame.astype(np.uint8), d, sigmaColor, sigmaSpace).astype(frame.dtype)
    
    elif method == 'nlm':  # Non-local means
        h = params.get('h', 10)
        temp = (normalize_for_display(frame) * 255).astype(np.uint8)
        result = cv2.fastNlMeansDenoising(
            temp, None, h=h, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        return result.astype(frame.dtype)
    
    return frame  # Return original if method not recognized

def create_contour_visualization(original_frame, processed_frame=None):
    """Create a visualization with original and processed frames plus contours"""
    # Calculate aspect ratio
    h, w = original_frame.shape
    aspect = w / h
    
    # Normalize for display
    norm_orig = normalize_for_display(original_frame)
    
    # Create original image and contour
    orig_img = hv.Image(norm_orig).options(
        title='Image Before',
        cmap='viridis',
        frame_width=400,
        aspect=aspect
    )
    
    orig_contour = hv.operation.contours(orig_img).options(
        title='Contours Before',
        cmap='viridis',
        frame_width=400,
        aspect=aspect
    )
    
    # If no processed frame is provided, just return original views
    if processed_frame is None:
        return pn.Row(orig_img, orig_contour)
    
    # Normalize processed frame
    norm_proc = normalize_for_display(processed_frame)
    
    # Create processed image and contour
    proc_img = hv.Image(norm_proc).options(
        title='Image After',
        cmap='viridis',
        frame_width=400,
        aspect=aspect
    )
    
    proc_contour = hv.operation.contours(proc_img).options(
        title='Contours After',
        cmap='viridis',
        frame_width=400,
        aspect=aspect
    )
    
    # Arrange in a grid: originals on top, processed on bottom
    layout = pn.Column(
        pn.Row(orig_img, orig_contour),
        pn.Row(proc_img, proc_contour)
    )
    
    return layout

# Function to run in the notebook
def run_denoising_with_contours(image_data, config=None, logger=None):
    """
    Run interactive denoising with contours visualization
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data with shape (frames, height, width)
    config : dict, optional
        Configuration dictionary to update with denoising parameters
    logger : logging.Logger, optional
        Logger object for error messages
    """
    if image_data is None:
        print("Please provide image data")
        return
    
    # Create parameters for the interactive app
    frame_selector = pn.widgets.IntSlider(
        name='Frame',
        start=0,
        end=min(image_data.shape[0]-1, 100),
        step=1, 
        value=0
    )
    
    method_selector = pn.widgets.Select(
        name='Method',
        options={
            'Gaussian': 'gaussian',
            'Median': 'median',
            'Bilateral': 'bilateral',
            'Non-local Means': 'nlm'
        },
        value='gaussian'
    )
    
    kernel_size = pn.widgets.IntSlider(
        name='Kernel Size',
        start=3,
        end=25,
        step=2,
        value=5
    )
    
    sigma = pn.widgets.FloatSlider(
        name='Sigma',
        start=0.1,
        end=10.0,
        step=0.1,
        value=1.5
    )
    
    sigma_color = pn.widgets.FloatSlider(
        name='Sigma Color',
        start=10,
        end=150,
        step=5,
        value=75
    )
    
    sigma_space = pn.widgets.FloatSlider(
        name='Sigma Space',
        start=10,
        end=150,
        step=5,
        value=75
    )
    
    h_param = pn.widgets.FloatSlider(
        name='h (Filter Strength)',
        start=1,
        end=30,
        step=1,
        value=10
    )
    
    # Parameter groups based on method
    gaussian_params = pn.Column(kernel_size, sigma)
    median_params = pn.Column(kernel_size)
    bilateral_params = pn.Column(kernel_size, sigma_color, sigma_space)
    nlm_params = pn.Column(h_param)
    
    param_container = pn.Column(gaussian_params)
    
    # Update parameters based on method selection
    def update_params(event):
        param_container.clear()
        if event.new == 'gaussian':
            param_container.append(gaussian_params)
        elif event.new == 'median':
            param_container.append(median_params)
        elif event.new == 'bilateral':
            param_container.append(bilateral_params)
        elif event.new == 'nlm':
            param_container.append(nlm_params)
    
    method_selector.param.watch(update_params, 'value')
    
    # Main function to update visualization
    @pn.depends(frame_selector, method_selector, kernel_size, sigma, sigma_color, sigma_space, h_param)
    def update_denoising(frame_idx, method, ksize, sigma_val, sigma_color_val, sigma_space_val, h_val):
        # Get the selected frame
        frame = image_data[frame_idx].copy()
        
        # Configure parameters based on method
        params = {}
        if method == 'gaussian':
            params = {'ksize': (ksize, ksize), 'sigma': sigma_val}
            title = f"Gaussian Denoised (k={ksize}, σ={sigma_val:.1f})"
        elif method == 'median':
            params = {'ksize': ksize}
            title = f"Median Denoised (k={ksize})"
        elif method == 'bilateral':
            params = {'d': ksize, 'sigmaColor': sigma_color_val, 'sigmaSpace': sigma_space_val}
            title = f"Bilateral Denoised (d={ksize}, σC={sigma_color_val:.1f}, σS={sigma_space_val:.1f})"
        elif method == 'nlm':
            params = {'h': h_val}
            title = f"Non-local Means Denoised (h={h_val:.1f})"
        
        # Process the frame
        import time
        start_time = time.time()
        
        try:
            # Apply denoising
            processed_frame = denoise_image(frame, method, **params)
            process_time = time.time() - start_time
            
            # Create visualization
            viz = create_contour_visualization(frame, processed_frame)
            
            # Add title and processing time
            header = pn.pane.Markdown(f"## {title}\nProcess time: {process_time:.3f}s")
            
            # Update config if available
            if config is not None:
                if 'preprocessing' not in config:
                    config['preprocessing'] = {}
                if 'denoise' not in config['preprocessing']:
                    config['preprocessing']['denoise'] = {}
                
                config['preprocessing']['denoise']['enabled'] = True
                config['preprocessing']['denoise']['method'] = method
                config['preprocessing']['denoise']['params'] = params
            
            return pn.Column(header, viz)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return pn.pane.Markdown(f"## Error\n{error_msg}")
    
    # Create app layout
    app = pn.Column(
        "## Advanced Denoising with Contours",
        pn.Row(
            pn.Column(frame_selector, method_selector, param_container, width=300),
            update_denoising
        )
    )
    
    display(app)