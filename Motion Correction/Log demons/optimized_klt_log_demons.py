import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize, warp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import cv2
import time

class OptimizedKLTLogDemonsCorrection:
    """
    Optimized KLT-Log-Demons motion correction implementation.
    Combines KLT feature tracking with simplified log-demons non-rigid registration.
    
    Parameters
    ----------
    max_iters : int, optional
        Maximum number of iterations for log-demons algorithm, by default 10
    sigma_fluid : float, optional
        Fluid regularization parameter, by default 2.0
    sigma_diffusion : float, optional
        Diffusion regularization parameter, by default 2.0
    step_size : float, optional
        Step size in log-demons algorithm, by default 1.0
    use_klt : bool, optional
        Whether to use KLT tracking for initial rigid correction, by default True
    klt_params : dict, optional
        Parameters for KLT tracking, by default settings tuned for calcium imaging
    batch_size : int, optional
        Number of frames to process in one batch, by default 100
    n_processes : int, optional
        Number of processes for parallel computation, by default 8
    downsample_factor : float, optional
        Factor to downsample images for faster processing, by default None (no downsampling)
    demons_every : int, optional
        Apply log-demons every N frames, interpolate between, by default 1
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default True
    verbose : bool, optional
        Whether to display progress information, by default True
    """
    
    def __init__(self, max_iters=10, sigma_fluid=2.0, sigma_diffusion=2.0, 
                 step_size=1.0, use_klt=True, klt_params=None, batch_size=100,
                 n_processes=8, downsample_factor=None, demons_every=1,
                 use_gpu=True, verbose=True):
        self.max_iters = max_iters
        self.sigma_fluid = sigma_fluid
        self.sigma_diffusion = sigma_diffusion
        self.step_size = step_size
        self.use_klt = use_klt
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.downsample_factor = downsample_factor
        self.demons_every = demons_every
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Check if GPU is available when requested
        if self.use_gpu and not self._check_gpu_available():
            print("Warning: GPU acceleration requested but not available. Falling back to CPU.")
            self.use_gpu = False
            
        # Default KLT parameters
        if klt_params is None:
            self.klt_params = {
                'maxCorners': 50,
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
        
        # For timing/benchmarking
        self.timing = {'klt': 0, 'demons': 0, 'total': 0}
        
    def _check_gpu_available(self):
        """Check if OpenCV has CUDA support"""
        try:
            cv2_build_info = cv2.getBuildInformation()
            return 'CUDA' in cv2_build_info and 'YES' in cv2_build_info.split('CUDA')[1].split('\n')[0]
        except:
            return False
    
    def _preprocess_frame(self, frame, smooth=True):
        """
        Preprocess a frame for registration.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
        smooth : bool, optional
            Whether to apply smoothing, by default True
            
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
            min_val = frame.min()
            max_val = frame.max()
            if max_val > min_val:
                frame = (frame - min_val) / (max_val - min_val)
            
        # Downsample if needed
        if self.downsample_factor is not None and self.downsample_factor != 1.0:
            new_shape = (int(frame.shape[0] / self.downsample_factor), 
                         int(frame.shape[1] / self.downsample_factor))
            frame = resize(frame, new_shape, anti_aliasing=True, preserve_range=True)
        
        # Apply Gaussian smoothing to reduce noise (optional)
        if smooth:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
        return frame
    
    def _track_features_klt(self, frame, template):
        """
        Track features using KLT and compute rigid transform.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Current frame
        template : numpy.ndarray
            Template frame
            
        Returns
        -------
        numpy.ndarray
            [dx, dy] translation vector
        """
        # Convert frames to 8-bit for OpenCV
        template_8bit = np.uint8(template * 255)
        frame_8bit = np.uint8(frame * 255)
        
        # Detect features in template
        corners = cv2.goodFeaturesToTrack(
            template_8bit, 
            self.klt_params['maxCorners'],
            self.klt_params['qualityLevel'],
            self.klt_params['minDistance'],
            blockSize=self.klt_params['blockSize']
        )
        
        if corners is None or len(corners) < 4:
            return np.zeros(2)
        
        # Track points
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            template_8bit, frame_8bit, 
            corners,
            None,
            winSize=self.klt_params['winSize'],
            maxLevel=self.klt_params['maxLevel'],
            criteria=self.klt_params['criteria']
        )
        
        # Keep only points that were successfully tracked
        if new_pts is None:
            return np.zeros(2)
        
        status = status.ravel().astype(bool)
        if not np.any(status):
            return np.zeros(2)
        
        prev_pts = corners[status].reshape(-1, 2)
        curr_pts = new_pts[status].reshape(-1, 2)
        
        # Estimate rigid transformation
        if len(prev_pts) >= 4:
            # Use RANSAC for robust estimation
            transform_matrix, inliers = cv2.estimateAffinePartial2D(
                prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            
            if transform_matrix is not None:
                # Extract translation part
                tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
                return np.array([tx, ty])
        
        # Fallback: use median of displacements
        displacements = curr_pts - prev_pts
        return np.median(displacements, axis=0)
    
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
        # Create transformation matrix
        matrix = np.array([[1, 0, -transform[0]], [0, 1, -transform[1]]])
        
        # Apply transformation using OpenCV
        corrected = cv2.warpAffine(
            frame, matrix, (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )
        
        return corrected
    
    def _compute_gradient(self, image):
        """
        Compute gradients of the image (using faster Sobel implementation).
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
            
        Returns
        -------
        tuple
            (dx, dy) gradients in x and y directions
        """
        # Use OpenCV's faster Sobel implementation
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return dx, dy
    
    def _simplified_demons_step(self, fixed, moving, v_field):
        """
        Perform one step of simplified log-demons algorithm.
        
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
        h, w = fixed.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Warp moving image with current velocity field
        warped_y = np.clip(y + v_field[0], 0, h-1)
        warped_x = np.clip(x + v_field[1], 0, w-1)
        
        # Use OpenCV's faster remap function instead of scipy's map_coordinates
        warped_coords = np.stack([warped_x, warped_y], axis=-1).astype(np.float32)
        warped_moving = cv2.remap(moving, warped_coords, None, cv2.INTER_LINEAR)
        
        # Compute difference and use pre-computed gradients for speed
        diff = fixed - warped_moving
        
        # Use averaged gradients as in demons algorithm
        dx_fixed, dy_fixed = self._compute_gradient(fixed)
        
        # Simplification: use fixed image gradients only (faster, slight accuracy tradeoff)
        denominator = dx_fixed**2 + dy_fixed**2 + 1e-6
        
        # Compute update force
        update_x = diff * dx_fixed / denominator
        update_y = diff * dy_fixed / denominator
        
        # Use OpenCV's faster filter for regularization
        update_x = cv2.GaussianBlur(update_x, (0, 0), self.sigma_fluid)
        update_y = cv2.GaussianBlur(update_y, (0, 0), self.sigma_fluid)
        
        # Update velocity field
        v_field[0] = v_field[0] + self.step_size * update_y
        v_field[1] = v_field[1] + self.step_size * update_x
        
        # Apply diffusion regularization
        v_field[0] = cv2.GaussianBlur(v_field[0], (0, 0), self.sigma_diffusion)
        v_field[1] = cv2.GaussianBlur(v_field[1], (0, 0), self.sigma_diffusion)
        
        return v_field
    
    def _log_demons_registration(self, fixed, moving, initial_transform=None):
        """
        Perform simplified log-demons registration between two frames.
        
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
        v_field = np.zeros((2, h, w), dtype=np.float32)
        
        # Apply initial rigid transform if provided
        if initial_transform is not None:
            v_field[0] = -initial_transform[1]  # y-direction
            v_field[1] = -initial_transform[0]  # x-direction
        
        # Iterative optimization (reduced number of iterations for speed)
        for i in range(self.max_iters):
            v_field = self._simplified_demons_step(fixed, moving, v_field)
        
        # Apply final velocity field
        y, x = np.mgrid[0:h, 0:w]
        warped_y = np.clip(y + v_field[0], 0, h-1).astype(np.float32)
        warped_x = np.clip(x + v_field[1], 0, w-1).astype(np.float32)
        
        # Use OpenCV for final warping (faster)
        warped_coords = np.stack([warped_x, warped_y], axis=-1)
        corrected = cv2.remap(moving, warped_coords, None, cv2.INTER_LINEAR)
        
        return v_field, corrected
    
    def _process_batch(self, frames, template, frame_indices):
        """
        Process a batch of frames.
        
        Parameters
        ----------
        frames : numpy.ndarray
            All input frames
        template : numpy.ndarray
            Template frame
        frame_indices : list
            Indices of frames to process
            
        Returns
        -------
        tuple
            (corrected_frames, velocity_fields, rigid_transforms)
        """
        n_frames = len(frame_indices)
        h, w = template.shape
        
        corrected_frames = np.zeros((n_frames, h, w), dtype=np.float32)
        velocity_fields = np.zeros((n_frames, 2, h, w), dtype=np.float32)
        rigid_transforms = np.zeros((n_frames, 2), dtype=np.float32)
        
        # Process frames with select frames for full demons and others just KLT
        for i, idx in enumerate(frame_indices):
            frame = frames[idx]
            
            # Step 1: KLT for rigid motion
            if self.use_klt:
                t_start = time.time()
                rigid_transform = self._track_features_klt(frame, template)
                frame_rigid = self._apply_rigid_transform(frame, rigid_transform)
                self.timing['klt'] += time.time() - t_start
            else:
                rigid_transform = np.zeros(2)
                frame_rigid = frame
                
            rigid_transforms[i] = rigid_transform
                
            # Step 2: Apply demons only on select frames for efficiency
            if self.demons_every > 1:
                # Apply demons only on select frames
                if i % self.demons_every == 0:
                    t_start = time.time()
                    v_field, corrected = self._log_demons_registration(
                        template, frame_rigid, None
                    )
                    self.timing['demons'] += time.time() - t_start
                    
                    corrected_frames[i] = corrected
                    velocity_fields[i] = v_field
                    
                    # Store for interpolation
                    last_demons_idx = i
                    last_v_field = v_field
                else:
                    # For frames between demons calculations, interpolate velocity field
                    if i > 0 and 'last_demons_idx' in locals():
                        # Distance-based interpolation weight
                        if (i - last_demons_idx) < self.demons_every:
                            # Linearly interpolate based on frame distance
                            alpha = (i - last_demons_idx) / self.demons_every
                            interp_v_field = (1 - alpha) * last_v_field
                            
                            # Apply interpolated field
                            y, x = np.mgrid[0:h, 0:w]
                            warped_y = np.clip(y + interp_v_field[0], 0, h-1).astype(np.float32)
                            warped_x = np.clip(x + interp_v_field[1], 0, w-1).astype(np.float32)
                            
                            warped_coords = np.stack([warped_x, warped_y], axis=-1)
                            corrected = cv2.remap(frame_rigid, warped_coords, None, cv2.INTER_LINEAR)
                            
                            corrected_frames[i] = corrected
                            velocity_fields[i] = interp_v_field
                        else:
                            # Too far from last demons, just use rigid correction
                            corrected_frames[i] = frame_rigid
                            velocity_fields[i] = np.zeros((2, h, w), dtype=np.float32)
                    else:
                        corrected_frames[i] = frame_rigid
                        velocity_fields[i] = np.zeros((2, h, w), dtype=np.float32)
            else:
                # Apply demons to every frame
                t_start = time.time()
                v_field, corrected = self._log_demons_registration(
                    template, frame_rigid, None
                )
                self.timing['demons'] += time.time() - t_start
                
                corrected_frames[i] = corrected
                velocity_fields[i] = v_field
                
        return corrected_frames, velocity_fields, rigid_transforms
    
    def correct_batch(self, frames, template_idx=0):
        """
        Apply KLT-log-demons correction to a batch of frames.
        
        Parameters
        ----------
        frames : numpy.ndarray
            Input frames with shape (t, h, w)
        template_idx : int, optional
            Index of the template frame, by default 0
            
        Returns
        -------
        tuple
            (corrected_frames, velocity_fields, rigid_transforms)
        """
        t_start_total = time.time()
        
        n_frames, h, w = frames.shape
        template_orig = frames[template_idx].copy()
        
        # Preprocess template
        template = self._preprocess_frame(template_orig)
        self.template = template
        
        # Initialize arrays for results
        if self.downsample_factor is not None and self.downsample_factor != 1.0:
            ds_h, ds_w = template.shape
            corrected_frames = np.zeros((n_frames, h, w), dtype=np.float32)
            velocity_fields = np.zeros((n_frames, 2, ds_h, ds_w), dtype=np.float32)
        else:
            corrected_frames = np.zeros_like(frames, dtype=np.float32)
            velocity_fields = np.zeros((n_frames, 2, h, w), dtype=np.float32)
            
        rigid_transforms = np.zeros((n_frames, 2), dtype=np.float32)
        
        # Copy template frame unchanged
        corrected_frames[template_idx] = template_orig
        
        # Preprocess all frames at once for efficiency
        preprocessed_frames = []
        for i in range(n_frames):
            if i != template_idx:
                preprocessed_frames.append(self._preprocess_frame(frames[i]))
        
        # Split into batches for processing
        all_indices = [i for i in range(n_frames) if i != template_idx]
        batches = [all_indices[i:i+self.batch_size] for i in range(0, len(all_indices), self.batch_size)]
        
        if self.verbose:
            print(f"Processing {n_frames} frames in {len(batches)} batches...")
            
        for batch_idx, batch_indices in enumerate(tqdm(batches, disable=not self.verbose)):
            # Get preprocessed frames for this batch
            batch_start = batch_idx * self.batch_size
            batch_frames = preprocessed_frames[batch_start:batch_start+len(batch_indices)]
            
            # Process batch
            batch_results = self._process_batch(
                {i: frame for i, frame in zip(batch_indices, batch_frames)}, 
                template, 
                batch_indices
            )
            
            # Collect results
            corrected_batch, velocity_batch, rigid_batch = batch_results
            
            # Handle upsampling if needed
            if (self.downsample_factor is not None and 
                self.downsample_factor != 1.0 and 
                corrected_batch.shape[1:] != (h, w)):
                
                # Upsample each corrected frame
                for i, idx in enumerate(batch_indices):
                    corrected_frames[idx] = resize(
                        corrected_batch[i], (h, w), 
                        preserve_range=True, anti_aliasing=True
                    )
                    
                    # Store downsampled velocity fields as is
                    velocity_fields[idx] = velocity_batch[i]
                    
                    # Scale rigid transforms according to downsample factor
                    rigid_transforms[idx] = rigid_batch[i] * self.downsample_factor
            else:
                for i, idx in enumerate(batch_indices):
                    corrected_frames[idx] = corrected_batch[i]
                    velocity_fields[idx] = velocity_batch[i]
                    rigid_transforms[idx] = rigid_batch[i]
        
        self.velocity_fields = velocity_fields
        self.displacements = rigid_transforms
        self.timing['total'] = time.time() - t_start_total
        
        if self.verbose:
            print(f"Total time: {self.timing['total']:.2f}s")
            print(f"KLT time: {self.timing['klt']:.2f}s")
            print(f"Demons time: {self.timing['demons']:.2f}s")
            
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
        
# Simplified usage function
def fast_motion_correct(video_data, template_idx=0, downsample_factor=2, 
                         demons_every=5, max_iters=5, output_dir=None, 
                         n_processes=8, save_results=False):
    """
    Apply optimized KLT-log-demons motion correction to a video.
    
    Parameters
    ----------
    video_data : numpy.ndarray
        Input video with shape (t, h, w)
    template_idx : int, optional
        Index of template frame, by default 0
    downsample_factor : float, optional
        Factor to downsample images for faster processing, by default 2
    demons_every : int, optional
        Apply log-demons every N frames, interpolate between, by default 5
    max_iters : int, optional
        Maximum iterations for log-demons, by default 5
    output_dir : str, optional
        Directory to save results, by default None
    n_processes : int, optional
        Number of processes for parallel processing, by default 8
    save_results : bool, optional
        Whether to save results, by default False
        
    Returns
    -------
    tuple
        (corrected_frames, velocity_fields, rigid_transforms)
    """
    # Create motion correction object with optimized parameters
    mc = OptimizedKLTLogDemonsCorrection(
        max_iters=max_iters,
        sigma_fluid=2.0,
        sigma_diffusion=2.0,
        step_size=1.0, 
        use_klt=True,
        n_processes=n_processes,
        downsample_factor=downsample_factor,
        demons_every=demons_every,
        batch_size=100,
        verbose=True
    )
    
    # Apply correction
    print(f"Starting optimized motion correction with:")
    print(f"  - Downsampling factor: {downsample_factor}x")
    print(f"  - Log-demons every {demons_every} frames")
    print(f"  - {max_iters} iterations per demons step")
    
    corrected_frames, velocity_fields, rigid_transforms = mc.correct_batch(
        video_data, template_idx=template_idx
    )
    
    # Save results if requested
    if save_results and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(output_dir, 'corrected_frames.npy'), corrected_frames)
        np.save(os.path.join(output_dir, 'rigid_transforms.npy'), rigid_transforms)
        
        # Visualize sample frames
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        n_frames = corrected_frames.shape[0]
        sample_frames = np.linspace(0, n_frames-1, min(5, n_frames), dtype=int)
        
        for idx in sample_frames:
            # Plot correction comparison
            fig = mc.plot_correction_comparison(
                video_data[idx], corrected_frames[idx], frame_idx=idx
            )
            fig.savefig(os.path.join(vis_dir, f'correction_comparison_frame_{idx}.png'))
            plt.close(fig)
    
    return corrected_frames, velocity_fields, rigid_transforms