import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import registration
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import warp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import cv2

class KLTLogDemonsCorrection:
    """
    KLT-Log-Demons motion correction implementation similar to Min1pipe.
    Combines KLT feature tracking with log-demons non-rigid registration.
    
    Parameters
    ----------
    max_iters : int, optional
        Maximum number of iterations for log-demons algorithm, by default 50
    sigma_fluid : float, optional
        Fluid regularization parameter, by default 2.0
    sigma_diffusion : float, optional
        Diffusion regularization parameter, by default 2.0
    sigma_i : float, optional
        Standard deviation for image smoothing, by default 1.0
    step_size : float, optional
        Step size in log-demons algorithm, by default 0.5
    use_klt : bool, optional
        Whether to use KLT tracking for initial rigid correction, by default True
    klt_params : dict, optional
        Parameters for KLT tracking, by default settings tuned for calcium imaging
    n_processes : int, optional
        Number of processes for parallel computation, by default 4
    verbose : bool, optional
        Whether to display progress information, by default True
    """
    
    def __init__(self, max_iters=50, sigma_fluid=2.0, sigma_diffusion=2.0, 
                 sigma_i=1.0, step_size=0.5, use_klt=True, 
                 klt_params=None, n_processes=4, verbose=True):
        self.max_iters = max_iters
        self.sigma_fluid = sigma_fluid
        self.sigma_diffusion = sigma_diffusion
        self.sigma_i = sigma_i
        self.step_size = step_size
        self.use_klt = use_klt
        self.n_processes = n_processes
        self.verbose = verbose
        
        # Default KLT parameters
        if klt_params is None:
            self.klt_params = {
                'maxCorners': 100,
                'qualityLevel': 0.01,
                'minDistance': 10,
                'blockSize': 5,
                'winSize': (15, 15),
                'maxLevel': 2,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }
        else:
            self.klt_params = klt_params
            
        # Initialize tracker
        self.feature_coords = None
        self.template = None
        self.displacements = None
        self.velocity_fields = None
        
    def _preprocess_frame(self, frame):
        """
        Preprocess a frame for registration.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
            
        Returns
        -------
        numpy.ndarray
            Preprocessed frame
        """
        # Ensure frame is float32 and normalized
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            
        # Apply Gaussian smoothing to reduce noise
        frame = gaussian_filter(frame, self.sigma_i)
        
        return frame
    
    def _detect_features(self, frame):
        """
        Detect good features to track using Shi-Tomasi algorithm.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
            
        Returns
        -------
        numpy.ndarray
            Detected feature coordinates
        """
        # Convert to 8-bit for OpenCV
        frame_8bit = np.uint8(frame * 255)
        
        # Detect features
        corners = cv2.goodFeaturesToTrack(
            frame_8bit, 
            self.klt_params['maxCorners'],
            self.klt_params['qualityLevel'],
            self.klt_params['minDistance'],
            blockSize=self.klt_params['blockSize']
        )
        
        if corners is None:
            # If no corners detected, return empty array
            return np.zeros((0, 2), dtype=np.float32)
        
        return corners.reshape(-1, 2)
    
    def _track_features(self, prev_frame, curr_frame, prev_pts):
        """
        Track features from previous frame to current frame using KLT.
        
        Parameters
        ----------
        prev_frame : numpy.ndarray
            Previous frame
        curr_frame : numpy.ndarray
            Current frame
        prev_pts : numpy.ndarray
            Feature points from previous frame
            
        Returns
        -------
        tuple
            (new_pts, status, err) where status indicates which points were successfully tracked
        """
        # Convert frames to 8-bit for OpenCV
        prev_8bit = np.uint8(prev_frame * 255)
        curr_8bit = np.uint8(curr_frame * 255)
        
        # Nothing to track
        if len(prev_pts) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32)
        
        # Reshape points for OpenCV if necessary
        if prev_pts.ndim == 2 and prev_pts.shape[1] == 2:
            prev_pts = prev_pts.reshape(-1, 1, 2)
        
        # Track points
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_8bit, curr_8bit, 
            prev_pts.astype(np.float32), 
            None,
            winSize=self.klt_params['winSize'],
            maxLevel=self.klt_params['maxLevel'],
            criteria=self.klt_params['criteria']
        )
        
        return new_pts, status, err
    
    def _estimate_rigid_transform(self, src_pts, dst_pts):
        """
        Estimate rigid transformation (translation) from point correspondences.
        
        Parameters
        ----------
        src_pts : numpy.ndarray
            Source points
        dst_pts : numpy.ndarray
            Destination points
            
        Returns
        -------
        numpy.ndarray
            Translation vector [dx, dy]
        """
        if len(src_pts) < 3 or len(dst_pts) < 3:
            return np.zeros(2)
        
        # Calculate the displacement of each point
        displacements = dst_pts - src_pts
        
        # Use RANSAC to filter outliers
        # Alternatively use median to be more robust to outliers
        translation = np.median(displacements, axis=0)
        
        return translation
    
    def _apply_rigid_transform(self, frame, transform):
        """
        Apply rigid transformation to a frame.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
        transform : numpy.ndarray
            Translation vector [dx, dy]
            
        Returns
        -------
        numpy.ndarray
            Transformed frame
        """
        # Create affine transformation matrix
        matrix = np.array([[1, 0, -transform[0]], [0, 1, -transform[1]]])
        
        # Apply transformation using OpenCV for better performance
        corrected = cv2.warpAffine(
            frame, matrix, (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )
        
        return corrected
    
    def _compute_gradient(self, image):
        """
        Compute gradients of the image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
            
        Returns
        -------
        tuple
            (dx, dy) gradients in x and y directions
        """
        dx = ndimage.sobel(image, axis=1)
        dy = ndimage.sobel(image, axis=0)
        return dx, dy
    
    def _demons_step(self, fixed, moving, v_field):
        """
        Perform one step of the log-demons algorithm.
        
        Parameters
        ----------
        fixed : numpy.ndarray
            Fixed (target) image
        moving : numpy.ndarray
            Moving (source) image
        v_field : numpy.ndarray
            Current velocity field
            
        Returns
        -------
        numpy.ndarray
            Updated velocity field
        """
        # Warp moving image with current velocity field
        h, w = fixed.shape
        y, x = np.mgrid[0:h, 0:w]
        points = np.stack((y, x), axis=0)
        
        warped_points = points + v_field
        warped_y, warped_x = warped_points[0], warped_points[1]
        
        # Interpolate
        warped_moving = ndimage.map_coordinates(moving, [warped_y, warped_x], order=1, mode='constant', cval=0)
        
        # Compute difference and gradients
        diff = fixed - warped_moving
        dx_fixed, dy_fixed = self._compute_gradient(fixed)
        dx_moving, dy_moving = self._compute_gradient(warped_moving)
        
        # Average gradients
        dx = (dx_fixed + dx_moving) / 2
        dy = (dy_fixed + dy_moving) / 2
        
        # Compute force
        denominator = dx**2 + dy**2 + 1e-8
        force_y = diff * dy / denominator
        force_x = diff * dx / denominator
        
        # Apply fluid-like regularization
        update_y = gaussian_filter(force_y, self.sigma_fluid)
        update_x = gaussian_filter(force_x, self.sigma_fluid)
        update = np.stack((update_y, update_x), axis=0)
        
        # Update velocity field
        v_field = v_field + self.step_size * update
        
        # Apply diffusion-like regularization
        v_field[0] = gaussian_filter(v_field[0], self.sigma_diffusion)
        v_field[1] = gaussian_filter(v_field[1], self.sigma_diffusion)
        
        return v_field
    
    def _log_demons_registration(self, fixed, moving, initial_transform=None):
        """
        Perform log-demons registration between two frames.
        
        Parameters
        ----------
        fixed : numpy.ndarray
            Fixed (target) image
        moving : numpy.ndarray
            Moving (source) image
        initial_transform : numpy.ndarray, optional
            Initial rigid transformation, by default None
            
        Returns
        -------
        tuple
            (velocity_field, corrected_frame)
        """
        h, w = fixed.shape
        
        # Initialize velocity field
        if initial_transform is not None:
            # Convert rigid transform to velocity field
            y, x = np.mgrid[0:h, 0:w]
            v_field = np.zeros((2, h, w), dtype=np.float32)
            v_field[0] = -initial_transform[1]  # y-direction
            v_field[1] = -initial_transform[0]  # x-direction
        else:
            v_field = np.zeros((2, h, w), dtype=np.float32)
        
        # Preprocess frames
        fixed_proc = self._preprocess_frame(fixed)
        moving_proc = self._preprocess_frame(moving)
        
        # Iterative optimization
        for i in range(self.max_iters):
            v_field = self._demons_step(fixed_proc, moving_proc, v_field)
        
        # Apply final transformation to original moving image
        y, x = np.mgrid[0:h, 0:w]
        warped_y = y + v_field[0]
        warped_x = x + v_field[1]
        
        corrected = ndimage.map_coordinates(
            moving, [warped_y, warped_x], 
            order=1, mode='constant', cval=0
        )
        
        return v_field, corrected
    
    def _process_frame(self, idx, frame, template):
        """
        Process a single frame with klt and log-demons.
        
        Parameters
        ----------
        idx : int
            Frame index
        frame : numpy.ndarray
            Frame to be processed
        template : numpy.ndarray
            Reference template
            
        Returns
        -------
        tuple
            (corrected_frame, velocity_field, rigid_transform)
        """
        rigid_transform = np.zeros(2)
        
        # First apply KLT for rigid correction if enabled
        if self.use_klt:
            # Preprocess frames
            template_proc = self._preprocess_frame(template)
            frame_proc = self._preprocess_frame(frame)
            
            # Detect features in template if not already detected
            if idx == 0 or self.feature_coords is None:
                self.feature_coords = self._detect_features(template_proc)
            
            # Track features from template to current frame
            if len(self.feature_coords) > 0:
                new_coords, status, _ = self._track_features(
                    template_proc, frame_proc, 
                    self.feature_coords.reshape(-1, 1, 2)
                )
                
                # Filter out points that couldn't be tracked
                if new_coords.shape[0] > 0:
                    status = status.ravel().astype(bool)
                    prev_pts = self.feature_coords.reshape(-1, 2)[status]
                    curr_pts = new_coords.reshape(-1, 2)[status]
                    
                    if len(prev_pts) >= 3:
                        # Estimate rigid transform
                        rigid_transform = self._estimate_rigid_transform(prev_pts, curr_pts)
                        
                        # Apply rigid transform
                        frame = self._apply_rigid_transform(frame, rigid_transform)
        
        # Next apply log-demons for non-rigid correction
        velocity_field, corrected_frame = self._log_demons_registration(
            template, frame, 
            initial_transform=None if self.use_klt else rigid_transform
        )
        
        return corrected_frame, velocity_field, rigid_transform
    
    def correct_batch(self, frames, template_idx=0, batch_size=None):
        """
        Apply KLT-log-demons correction to a batch of frames.
        
        Parameters
        ----------
        frames : numpy.ndarray
            Input frames with shape (t, h, w)
        template_idx : int, optional
            Index of the template frame, by default 0
        batch_size : int, optional
            Process frames in batches of this size, by default None (all frames)
            
        Returns
        -------
        tuple
            (corrected_frames, velocity_fields, rigid_transforms)
        """
        n_frames, h, w = frames.shape
        self.template = frames[template_idx].copy()
        
        # Initialize arrays to store results
        corrected_frames = np.zeros_like(frames)
        velocity_fields = np.zeros((n_frames, 2, h, w), dtype=np.float32)
        rigid_transforms = np.zeros((n_frames, 2), dtype=np.float32)
        
        # Add the template frame unchanged
        corrected_frames[template_idx] = self.template
        
        # Process frames in parallel
        if batch_size is None:
            batch_size = n_frames
        
        for batch_start in tqdm(range(0, n_frames, batch_size), disable=not self.verbose):
            batch_end = min(batch_start + batch_size, n_frames)
            batch_indices = list(range(batch_start, batch_end))
            
            # Skip template frame
            if template_idx in batch_indices:
                batch_indices.remove(template_idx)
            
            if self.n_processes <= 1:
                # Process sequentially
                for idx in tqdm(batch_indices, disable=not self.verbose):
                    corrected, v_field, r_transform = self._process_frame(
                        idx, frames[idx], self.template
                    )
                    corrected_frames[idx] = corrected
                    velocity_fields[idx] = v_field
                    rigid_transforms[idx] = r_transform
            else:
                # Process in parallel
                with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                    results = list(executor.map(
                        self._process_frame,
                        batch_indices,
                        [frames[i] for i in batch_indices],
                        [self.template] * len(batch_indices)
                    ))
                
                # Collect results
                for i, idx in enumerate(batch_indices):
                    corrected_frames[idx], velocity_fields[idx], rigid_transforms[idx] = results[i]
        
        self.velocity_fields = velocity_fields
        self.displacements = rigid_transforms
        
        return corrected_frames, velocity_fields, rigid_transforms
    
    def plot_displacement_field(self, velocity_field, frame_idx=None, subsample=10):
        """
        Plot displacement field from the log-demons registration.
        
        Parameters
        ----------
        velocity_field : numpy.ndarray
            Velocity field with shape (2, h, w)
        frame_idx : int, optional
            Frame index for title, by default None
        subsample : int, optional
            Subsample factor for displacement field, by default 10
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with displacement field visualization
        """
        h, w = velocity_field.shape[1:3]
        
        # Create a grid of coordinates
        y, x = np.mgrid[0:h:subsample, 0:w:subsample]
        
        # Subsample the velocity field
        v_y = velocity_field[0, ::subsample, ::subsample]
        v_x = velocity_field[1, ::subsample, ::subsample]
        
        # Compute displacement magnitude
        magnitude = np.sqrt(v_y**2 + v_x**2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot displacement field as color-coded arrows
        quiv = ax.quiver(x, y, v_x, v_y, magnitude, 
                         cmap='jet', scale=1, scale_units='xy')
        plt.colorbar(quiv, ax=ax, label='Displacement Magnitude')
        
        # Add title
        title = "Displacement Field"
        if frame_idx is not None:
            title += f" - Frame {frame_idx}"
        ax.set_title(title)
        
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Invert y-axis to match image coordinates
        ax.set_aspect('equal')
        
        return fig
    
    def plot_correction_comparison(self, original, corrected, frame_idx=None):
        """
        Plot comparison between original and corrected frames.
        
        Parameters
        ----------
        original : numpy.ndarray
            Original frame
        corrected : numpy.ndarray
            Corrected frame
        frame_idx : int, optional
            Frame index for title, by default None
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with comparison visualization
        """
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original frame
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Frame")
        axes[0].axis('off')
        
        # Plot corrected frame
        axes[1].imshow(corrected, cmap='gray')
        axes[1].set_title("Corrected Frame")
        axes[1].axis('off')
        
        # Plot difference
        diff = original - corrected
        vmax = np.max([np.abs(diff.min()), np.abs(diff.max())])
        im = axes[2].imshow(diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[2].set_title("Difference")
        axes[2].axis('off')
        
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add main title
        title = "Motion Correction Comparison"
        if frame_idx is not None:
            title += f" - Frame {frame_idx}"
        fig.suptitle(title)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, output_dir, corrected_frames, velocity_fields=None, 
                     original_frames=None, visualize=True, save_npy=True):
        """
        Save correction results and optional visualizations.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        corrected_frames : numpy.ndarray
            Corrected frames with shape (t, h, w)
        velocity_fields : numpy.ndarray, optional
            Velocity fields with shape (t, 2, h, w), by default None
        original_frames : numpy.ndarray, optional
            Original frames for comparison, by default None
        visualize : bool, optional
            Whether to save visualizations, by default True
        save_npy : bool, optional
            Whether to save results as .npy files, by default True
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save corrected frames as .npy
        if save_npy:
            np.save(os.path.join(output_dir, 'corrected_frames.npy'), corrected_frames)
            
            if velocity_fields is not None:
                np.save(os.path.join(output_dir, 'velocity_fields.npy'), velocity_fields)
        
        # Save visualizations
        if visualize:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save sample frames
            n_frames = corrected_frames.shape[0]
            sample_frames = np.linspace(0, n_frames-1, min(10, n_frames), dtype=int)
            
            for idx in sample_frames:
                # Plot displacement field
                if velocity_fields is not None:
                    fig = self.plot_displacement_field(velocity_fields[idx], frame_idx=idx)
                    fig.savefig(os.path.join(vis_dir, f'displacement_field_frame_{idx}.png'))
                    plt.close(fig)
                
                # Plot correction comparison
                if original_frames is not None:
                    fig = self.plot_correction_comparison(
                        original_frames[idx], corrected_frames[idx], frame_idx=idx
                    )
                    fig.savefig(os.path.join(vis_dir, f'correction_comparison_frame_{idx}.png'))
                    plt.close(fig)

# Example usage function
def motion_correct_video(video_data, template_idx=0, max_iters=50, sigma_fluid=2.0,
                         sigma_diffusion=2.0, use_klt=True, n_processes=4,
                         output_dir=None, save_results=False):
    """
    Apply KLT-log-demons motion correction to a video.
    
    Parameters
    ----------
    video_data : numpy.ndarray
        Input video with shape (t, h, w)
    template_idx : int, optional
        Index of template frame, by default 0
    max_iters : int, optional
        Maximum iterations for log-demons, by default 50
    sigma_fluid : float, optional
        Fluid regularization parameter, by default 2.0
    sigma_diffusion : float, optional
        Diffusion regularization parameter, by default 2.0
    use_klt : bool, optional
        Whether to use KLT for initial correction, by default True
    n_processes : int, optional
        Number of processes for parallel processing, by default 4
    output_dir : str, optional
        Directory to save results, by default None
    save_results : bool, optional
        Whether to save results, by default False
        
    Returns
    -------
    tuple
        (corrected_frames, velocity_fields, rigid_transforms)
    """
    # Create motion correction object
    mc = KLTLogDemonsCorrection(
        max_iters=max_iters,
        sigma_fluid=sigma_fluid,
        sigma_diffusion=sigma_diffusion,
        use_klt=use_klt,
        n_processes=n_processes,
        verbose=True
    )
    
    # Apply correction
    corrected_frames, velocity_fields, rigid_transforms = mc.correct_batch(
        video_data, template_idx=template_idx
    )
    
    # Save results if requested
    if save_results and output_dir is not None:
        mc.save_results(
            output_dir,
            corrected_frames,
            velocity_fields,
            original_frames=video_data,
            visualize=True,
            save_npy=True
        )
    
    return corrected_frames, velocity_fields, rigid_transforms