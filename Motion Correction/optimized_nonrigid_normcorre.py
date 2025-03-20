import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import time
import psutil
from functools import partial
from scipy import ndimage
from skimage.transform import resize
from tifffile import imread, imwrite

class OptimizedNonRigidCorrection:
    """
    Optimized implementation of non-rigid NormCorre motion correction.
    
    Parameters
    ----------
    max_shifts : tuple or list, optional
        Maximum allowed shifts in pixels (y, x), by default (20, 20)
    patch_size : tuple or list, optional
        Size of patches for non-rigid registration (y, x), by default (48, 48)
    stride : tuple or list, optional
        Stride between patches (y, x), by default (24, 24)
    grid_size : tuple or list, optional
        Grid size for non-rigid registration (y, x), by default (6, 6)
    overlap_factor : float, optional
        Factor for patch overlap, by default 0.5
    iterations : int, optional
        Number of iterations for refinement, by default 2
    n_processes : int, optional
        Number of processes for parallel computation, by default 8
    batch_size : int, optional
        Number of frames to process in one batch, by default 100
    upsample_factor : int, optional
        Upsample factor for subpixel registration, by default 1
    timeout : int, optional
        Timeout in seconds for processing a single frame, by default 30
    highpass_sigma : float, optional
        Sigma for highpass filter, by default 1.5
    min_patch_variance : float, optional
        Minimum variance for a patch to be considered valid, by default 0.001
    fill_fraction : float, optional
        Fraction of overlap region to consider for interpolation, by default 0.5
    verbose : bool, optional
        Whether to display progress information, by default True
    """
    def __init__(self, max_shifts=(20, 20), patch_size=(48, 48), stride=None, 
                 grid_size=(6, 6), overlap_factor=0.5, iterations=2,
                 n_processes=8, batch_size=100, upsample_factor=1,
                 timeout=30, highpass_sigma=1.5, min_patch_variance=0.001, 
                 fill_fraction=0.5, verbose=True):
        
        self.max_shifts = max_shifts
        self.patch_size = patch_size
        self.stride = stride if stride is not None else (int(patch_size[0]*overlap_factor), 
                                                         int(patch_size[1]*overlap_factor))
        self.grid_size = grid_size
        self.iterations = iterations
        self.n_processes = min(n_processes, os.cpu_count())
        self.batch_size = batch_size
        self.upsample_factor = upsample_factor
        self.timeout = timeout
        self.highpass_sigma = highpass_sigma
        self.min_patch_variance = min_patch_variance
        self.fill_fraction = fill_fraction
        self.verbose = verbose
        
        # Adjust processes based on available memory
        mem = psutil.virtual_memory()
        if mem.available < 4 * 1024 * 1024 * 1024:  # Less than 4GB available
            self.n_processes = min(self.n_processes, 4)
            if self.verbose:
                print(f"Limited processes to {self.n_processes} due to available memory")
        
        # Initialize results
        self.rigid_shifts = None
        self.nonrigid_shifts = None
        self.template = None
        
    def correct(self, video_data, template=None, rigid_first=True):
        """
        Apply motion correction to a video.
        
        Parameters
        ----------
        video_data : numpy.ndarray
            Input video data with shape (t, y, x)
        template : numpy.ndarray, optional
            Custom template for registration, by default None
        rigid_first : bool, optional
            Whether to apply rigid correction first, by default True
            
        Returns
        -------
        tuple
            (corrected_video, shifts_dict, template)
        """
        start_time = time.time()
        
        T, h, w = video_data.shape
        
        # Create template if not provided
        if template is None:
            template = self._create_template(video_data)
        self.template = template
        
        # Initialize results
        corrected_video = np.zeros_like(video_data)
        
        # Apply rigid registration first if requested
        rigid_shifts = None
        if rigid_first:
            if self.verbose:
                print("Applying rigid motion correction...")
                
            corrected_video, rigid_shifts = self._apply_rigid_correction(video_data, template)
            
            # Update template after rigid correction
            n_frames_template = min(100, T)
            frames_for_template = np.arange(0, T, max(1, T//n_frames_template))[:n_frames_template]
            template = np.mean(corrected_video[frames_for_template], axis=0)
            self.template = template
        else:
            corrected_video = video_data.copy()
        
        # Apply non-rigid registration
        if self.verbose:
            print("Applying non-rigid motion correction...")
            
        corrected_video, nonrigid_shifts = self._apply_nonrigid_correction(corrected_video, template)
        
        # Combine shift information
        shifts_dict = {
            'rigid': rigid_shifts,
            'nonrigid': nonrigid_shifts
        }
        
        self.rigid_shifts = rigid_shifts
        self.nonrigid_shifts = nonrigid_shifts
        
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Motion correction completed in {total_time:.2f} seconds")
        
        return corrected_video, shifts_dict, template
    
    def _create_template(self, video_data):
        """
        Create template for registration.
        
        Parameters
        ----------
        video_data : numpy.ndarray
            Input video data
            
        Returns
        -------
        numpy.ndarray
            Template for registration
        """
        T = video_data.shape[0]
        
        # Use a subset of frames for template
        n_template_frames = min(100, T)
        interval = max(1, T // n_template_frames)
        template_idx = np.arange(0, T, interval)[:n_template_frames]
        
        if self.verbose:
            print(f"Creating template from {len(template_idx)} frames...")
        
        # Create median image from selected frames (more robust than mean)
        template = np.median(video_data[template_idx], axis=0)
        
        # Apply high-pass filter to enhance features
        template = self._preprocess_image(template)
        
        return template
    
    def _preprocess_image(self, image):
        """
        Preprocess image for registration.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            Preprocessed image
        """
        # Handle NaNs
        image = np.nan_to_num(image)
        
        # High-pass filter to enhance features
        image = image.astype(np.float32)
        blurred = cv2.GaussianBlur(image, (0, 0), self.highpass_sigma)
        image = image - blurred
        
        return image
    
    def _apply_rigid_correction(self, video_data, template):
        """
        Apply rigid motion correction.
        
        Parameters
        ----------
        video_data : numpy.ndarray
            Input video data
        template : numpy.ndarray
            Template for registration
            
        Returns
        -------
        tuple
            (corrected_video, shifts)
        """
        T, h, w = video_data.shape
        
        # Preprocess template
        template_proc = self._preprocess_image(template)
        
        # Initialize results
        corrected_video = np.zeros_like(video_data)
        shifts = np.zeros((T, 2))
        
        # Process in batches
        batches = [(i, min(i + self.batch_size, T)) 
                   for i in range(0, T, self.batch_size)]
        
        for batch_idx, (start, end) in enumerate(batches):
            if self.verbose:
                print(f"Rigid correction batch {batch_idx+1}/{len(batches)} (frames {start}-{end-1})...")
            
            # Extract current batch
            batch_frames = video_data[start:end]
            batch_size = end - start
            
            # Process frames in parallel
            if self.n_processes > 1 and batch_size > 1:
                # Define a partial function for parallel processing
                process_func = partial(self._process_frame_rigid, template=template_proc)
                
                with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                    # Process all frames in the batch
                    results = list(tqdm(
                        executor.map(process_func, batch_frames), 
                        total=batch_size,
                        disable=not self.verbose
                    ))
                    
                    # Extract results
                    for i, (corrected_frame, shift) in enumerate(results):
                        corrected_video[start + i] = corrected_frame
                        shifts[start + i] = shift
            else:
                # Serial processing
                for i in tqdm(range(batch_size), disable=not self.verbose):
                    corrected_frame, shift = self._process_frame_rigid(
                        batch_frames[i], template_proc
                    )
                    corrected_video[start + i] = corrected_frame
                    shifts[start + i] = shift
        
        return corrected_video, shifts
    
    def _process_frame_rigid(self, frame, template):
        """
        Process a single frame for rigid correction.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
        template : numpy.ndarray
            Template for registration
            
        Returns
        -------
        tuple
            (corrected_frame, shift)
        """
        # Preprocess frame
        frame_proc = self._preprocess_image(frame)
        
        # Calculate shift
        shift = self._compute_shift(template, frame_proc)
        
        # Apply shift
        corrected_frame = self._apply_shift(frame, shift)
        
        return corrected_frame, shift
    
    def _compute_shift(self, template, frame):
        """
        Compute shift between template and frame.
        
        Parameters
        ----------
        template : numpy.ndarray
            Template image
        frame : numpy.ndarray
            Frame to register
            
        Returns
        -------
        numpy.ndarray
            [y_shift, x_shift]
        """
        max_shift_y, max_shift_x = self.max_shifts
        
        # Calculate using FFT-based approach for speed
        result = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
        
        # Find maximum
        _, _, _, max_loc = cv2.minMaxLoc(result)
        y_shift, x_shift = max_loc[1] - template.shape[0]//2, max_loc[0] - template.shape[1]//2
        
        # Apply subpixel refinement if requested
        if self.upsample_factor > 1:
            # Extract region around peak
            region_size = 3
            y_start = max(0, max_loc[1] - region_size)
            y_end = min(result.shape[0], max_loc[1] + region_size + 1)
            x_start = max(0, max_loc[0] - region_size)
            x_end = min(result.shape[1], max_loc[0] + region_size + 1)
            region = result[y_start:y_end, x_start:x_end]
            
            # Upsample region
            upsampled = cv2.resize(region, None, fx=self.upsample_factor, fy=self.upsample_factor, 
                                   interpolation=cv2.INTER_CUBIC)
            
            # Find peak in upsampled region
            _, _, _, up_max_loc = cv2.minMaxLoc(upsampled)
            
            # Calculate refined shift
            y_shift = (y_start + up_max_loc[1] / self.upsample_factor - template.shape[0]//2)
            x_shift = (x_start + up_max_loc[0] / self.upsample_factor - template.shape[1]//2)
        
        # Limit to max shifts
        y_shift = np.clip(y_shift, -max_shift_y, max_shift_y)
        x_shift = np.clip(x_shift, -max_shift_x, max_shift_x)
        
        return np.array([-y_shift, -x_shift])  # Negative because we want to move frame to match template
    
    def _apply_shift(self, frame, shift):
        """
        Apply shift to frame.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
        shift : numpy.ndarray
            Shift to apply [y_shift, x_shift]
            
        Returns
        -------
        numpy.ndarray
            Shifted frame
        """
        # Create transformation matrix
        M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
        
        # Apply transformation
        corrected = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        
        return corrected
    
    def _apply_nonrigid_correction(self, video_data, template):
        """
        Apply non-rigid motion correction.
        
        Parameters
        ----------
        video_data : numpy.ndarray
            Input video data (already rigid-corrected if rigid_first=True)
        template : numpy.ndarray
            Template for registration
            
        Returns
        -------
        tuple
            (corrected_video, shifts)
        """
        T, h, w = video_data.shape
        
        # Calculate patch coordinates
        patch_coords = self._get_patch_grid(h, w)
        n_patches = len(patch_coords)
        
        # Preprocess template once
        template_proc = self._preprocess_image(template)
        
        # Extract template patches
        template_patches = self._extract_patches(template_proc, patch_coords)
        
        # Initialize results
        corrected_video = np.zeros_like(video_data)
        nonrigid_shifts = np.zeros((T, n_patches, 2))
        
        # Process in batches
        batches = [(i, min(i + self.batch_size, T)) 
                   for i in range(0, T, self.batch_size)]
        
        for batch_idx, (start, end) in enumerate(batches):
            if self.verbose:
                print(f"Non-rigid correction batch {batch_idx+1}/{len(batches)} "
                      f"(frames {start}-{end-1}, {n_patches} patches per frame)...")
            
            # Extract current batch
            batch_frames = video_data[start:end]
            batch_size = end - start
            
            # Process frames in smaller chunks to avoid memory issues
            chunk_size = max(1, self.n_processes // 2)  # Process fewer frames at once
            chunks = [(i, min(i + chunk_size, batch_size)) 
                      for i in range(0, batch_size, chunk_size)]
            
            for chunk_start, chunk_end in chunks:
                frames_to_process = batch_frames[chunk_start:chunk_end]
                
                # Process frames in parallel
                if self.n_processes > 1 and len(frames_to_process) > 1:
                    # Define process function with fixed parameters
                    process_func = partial(
                        self._process_frame_nonrigid,
                        template_patches=template_patches,
                        patch_coords=patch_coords,
                        img_shape=(h, w)
                    )
                    
                    with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                        # Submit tasks with timeouts to prevent hanging
                        futures = [
                            executor.submit(process_func, frame) 
                            for frame in frames_to_process
                        ]
                        
                        # Process results as they complete
                        for i, future in enumerate(as_completed(futures)):
                            try:
                                idx = chunk_start + i
                                corrected, shifts = future.result(timeout=self.timeout)
                                corrected_video[start + idx] = corrected
                                nonrigid_shifts[start + idx] = shifts
                            except TimeoutError:
                                if self.verbose:
                                    print(f"Frame {start + chunk_start + i} timed out, using rigid approach")
                                # Fall back to rigid correction for this frame
                                frame = frames_to_process[i]
                                frame_proc = self._preprocess_image(frame)
                                shift = self._compute_shift(template_proc, frame_proc)
                                corrected = self._apply_shift(frame, shift)
                                corrected_video[start + chunk_start + i] = corrected
                                # Set all patch shifts to the rigid shift
                                nonrigid_shifts[start + chunk_start + i] = np.tile(shift, (n_patches, 1))
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error processing frame {start + chunk_start + i}: {str(e)}")
                                # Use original frame on error
                                corrected_video[start + chunk_start + i] = frames_to_process[i]
                else:
                    # Serial processing
                    for i, frame in enumerate(frames_to_process):
                        idx = start + chunk_start + i
                        try:
                            corrected, shifts = self._process_frame_nonrigid(
                                frame, template_patches, patch_coords, (h, w)
                            )
                            corrected_video[idx] = corrected
                            nonrigid_shifts[idx] = shifts
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing frame {idx}: {str(e)}")
                            # Fall back to rigid correction
                            frame_proc = self._preprocess_image(frame)
                            shift = self._compute_shift(template_proc, frame_proc)
                            corrected = self._apply_shift(frame, shift)
                            corrected_video[idx] = corrected
                            nonrigid_shifts[idx] = np.tile(shift, (n_patches, 1))
        
        return corrected_video, nonrigid_shifts
    
    def _get_patch_grid(self, h, w):
        """
        Calculate patch grid coordinates.
        
        Parameters
        ----------
        h : int
            Image height
        w : int
            Image width
            
        Returns
        -------
        list
            List of (y, x, h, w) patch coordinates
        """
        # If grid_size is specified, create a grid with that many patches
        if self.grid_size is not None:
            grid_h, grid_w = self.grid_size
            patches = []
            
            # Calculate patch positions based on grid size
            for i in range(grid_h):
                for j in range(grid_w):
                    # Calculate patch position ensuring coverage of the entire image
                    y = i * (h - self.patch_size[0]) // max(1, grid_h - 1)
                    x = j * (w - self.patch_size[1]) // max(1, grid_w - 1)
                    
                    # Ensure patches stay within image bounds
                    y = min(y, h - self.patch_size[0])
                    x = min(x, w - self.patch_size[1])
                    
                    patches.append((y, x, self.patch_size[0], self.patch_size[1]))
            
            return patches
        
        # Otherwise, create overlapping patches based on stride
        y_positions = list(range(0, h - self.patch_size[0] + 1, self.stride[0]))
        x_positions = list(range(0, w - self.patch_size[1] + 1, self.stride[1]))
        
        # Ensure we cover the entire image
        if y_positions[-1] + self.patch_size[0] < h:
            y_positions.append(h - self.patch_size[0])
        if x_positions[-1] + self.patch_size[1] < w:
            x_positions.append(w - self.patch_size[1])
        
        patches = []
        for y in y_positions:
            for x in x_positions:
                patches.append((y, x, self.patch_size[0], self.patch_size[1]))
        
        return patches
    
    def _extract_patches(self, image, patch_coords):
        """
        Extract patches from an image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
        patch_coords : list
            List of (y, x, h, w) patch coordinates
            
        Returns
        -------
        list
            List of patches
        """
        patches = []
        for y, x, h, w in patch_coords:
            patch = image[y:y+h, x:x+w]
            patches.append(patch)
        return patches
    
    def _process_frame_nonrigid(self, frame, template_patches, patch_coords, img_shape):
        """
        Process a single frame for non-rigid correction.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
        template_patches : list
            List of template patches
        patch_coords : list
            List of (y, x, h, w) patch coordinates
        img_shape : tuple
            (height, width) of the image
            
        Returns
        -------
        tuple
            (corrected_frame, shifts)
        """
        h, w = img_shape
        n_patches = len(patch_coords)
        
        # Preprocess frame
        frame_proc = self._preprocess_image(frame)
        
        # Calculate shifts for each patch
        patch_shifts = np.zeros((n_patches, 2))
        
        for i, ((y, x, ph, pw), template_patch) in enumerate(zip(patch_coords, template_patches)):
            # Extract frame patch
            frame_patch = frame_proc[y:y+ph, x:x+pw]
            
            # Skip patches with low variance (likely background)
            if np.var(frame_patch) < self.min_patch_variance:
                continue
                
            # Calculate shift for this patch
            patch_shift = self._compute_shift(template_patch, frame_patch)
            patch_shifts[i] = patch_shift
        
        # Apply piecewise rigid shift using interpolation
        corrected_frame = self._apply_nonrigid_shift(frame, patch_shifts, patch_coords, (h, w))
        
        return corrected_frame, patch_shifts
    
    def _apply_nonrigid_shift(self, frame, shifts, patch_coords, img_shape):
        """
        Apply non-rigid shift to a frame.
        
        Parameters
        ----------
        frame : numpy.ndarray
            Input frame
        shifts : numpy.ndarray
            Shifts for each patch, shape (n_patches, 2)
        patch_coords : list
            List of (y, x, h, w) patch coordinates
        img_shape : tuple
            (height, width) of the image
            
        Returns
        -------
        numpy.ndarray
            Corrected frame
        """
        h, w = img_shape
        
        # Create displacement field
        flow = np.zeros((h, w, 2), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        # Add displacement from each patch
        for (y, x, ph, pw), (shift_y, shift_x) in zip(patch_coords, shifts):
            # Create weight mask for this patch (higher in center, lower at edges)
            y_coords, x_coords = np.mgrid[y:y+ph, x:x+pw]
            
            # Distance from patch center (normalized to [0,1])
            center_y, center_x = y + ph//2, x + pw//2
            dist_y = 1.0 - np.abs(y_coords - center_y) / (ph//2 + 1e-10)
            dist_x = 1.0 - np.abs(x_coords - center_x) / (pw//2 + 1e-10)
            
            # Combine to get weight (higher in center)
            weight = np.clip(dist_y * dist_x, 0, 1) ** 2
            
            # Add weighted shift
            flow[y:y+ph, x:x+pw, 0] += shift_y * weight
            flow[y:y+ph, x:x+pw, 1] += shift_x * weight
            weights[y:y+ph, x:x+pw] += weight
        
        # Normalize by weights
        valid = weights > 0
        flow[valid, 0] /= weights[valid]
        flow[valid, 1] /= weights[valid]
        
        # Create mesh grid
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Apply displacement field
        y_shifted = np.clip(y_coords + flow[:, :, 0], 0, h-1)
        x_shifted = np.clip(x_coords + flow[:, :, 1], 0, w-1)
        
        # Remap using OpenCV for efficiency
        map_x = x_shifted.astype(np.float32)
        map_y = y_shifted.astype(np.float32)
        
        corrected = cv2.remap(frame, map_x, map_y, 
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
        
        return corrected
    
    def visualize_shifts(self, shifts_dict, frame_idx=None):
        """
        Visualize shifts.
        
        Parameters
        ----------
        shifts_dict : dict
            Dictionary with 'rigid' and 'nonrigid' shifts
        frame_idx : int, optional
            Frame index, by default None
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with visualization
        """
        if frame_idx is None:
            # Show summary of all frames
            return self._visualize_shift_summary(shifts_dict)
        else:
            # Show specific frame
            return self._visualize_frame_shifts(shifts_dict, frame_idx)
    
    def _visualize_shift_summary(self, shifts_dict):
        """
        Visualize summary of shifts.
        
        Parameters
        ----------
        shifts_dict : dict
            Dictionary with 'rigid' and 'nonrigid' shifts
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with visualization
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot rigid shifts
        if shifts_dict['rigid'] is not None:
            rigid_shifts = shifts_dict['rigid']
            frames = np.arange(len(rigid_shifts))
            
            # Plot y shifts
            axes[0].plot(frames, rigid_shifts[:, 0], 'b-', label='Y')
            
            # Plot x shifts
            axes[0].plot(frames, rigid_shifts[:, 1], 'r-', label='X')
            
            axes[0].set_ylabel('Shift (pixels)')
            axes[0].set_title('Rigid Shifts')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot nonrigid shift magnitudes
        if shifts_dict['nonrigid'] is not None:
            nonrigid_shifts = shifts_dict['nonrigid']
            frames = np.arange(len(nonrigid_shifts))
            
            # Calculate mean magnitude for each frame
            mean_magnitudes = np.mean(
                np.sqrt(nonrigid_shifts[:, :, 0]**2 + nonrigid_shifts[:, :, 1]**2),
                axis=1
            )
            
            # Plot mean magnitudes
            axes[1].plot(frames, mean_magnitudes)
            axes[1].set_xlabel('Frame')
            axes[1].set_ylabel('Mean Shift Magnitude (pixels)')
            axes[1].set_title('Non-rigid Shift Magnitudes')
            axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def _visualize_frame_shifts(self, shifts_dict, frame_idx):
        """
        Visualize shifts for a specific frame.
        
        Parameters
        ----------
        shifts_dict : dict
            Dictionary with 'rigid' and 'nonrigid' shifts
        frame_idx : int
            Frame index
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with visualization
        """
        if shifts_dict['nonrigid'] is not None:
            # Visualize non-rigid shifts
            nonrigid_shifts = shifts_dict['nonrigid'][frame_idx]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create a regular grid for visualization
            n_patches = nonrigid_shifts.shape[0]
            grid_size = int(np.sqrt(n_patches))
            grid_x, grid_y = np.meshgrid(range(grid_size), range(grid_size))
            
            # Reshape shifts for grid visualization
            grid_shifts = nonrigid_shifts[:grid_size*grid_size].reshape(grid_size, grid_size, 2)
            
            # Plot shifts as quiver
            q = ax.quiver(grid_x, grid_y, 
                          grid_shifts[:, :, 1], grid_shifts[:, :, 0],  # x and y shifts
                          np.sqrt(grid_shifts[:, :, 0]**2 + grid_shifts[:, :, 1]**2),  # magnitude
                          cmap='viridis',
                          scale=5,
                          angles='xy', 
                          scale_units='xy')
            
            plt.colorbar(q, ax=ax, label='Shift Magnitude (pixels)')
            ax.set_title(f'Non-rigid Shifts for Frame {frame_idx}')
            ax.set_xlabel('X Grid Point')
            ax.set_ylabel('Y Grid Point')
            ax.invert_yaxis()  # To match image coordinates
            
            return fig
        elif shifts_dict['rigid'] is not None:
            # Just show rigid shift
            fig, ax = plt.subplots(figsize=(8, 6))
            shift = shifts_dict['rigid'][frame_idx]
            ax.arrow(0, 0, shift[1], shift[0], head_width=0.2, head_length=0.3, fc='blue', ec='blue')
            ax.set_xlim(-max(1, abs(shift[1])*1.5), max(1, abs(shift[1])*1.5))
            ax.set_ylim(-max(1, abs(shift[0])*1.5), max(1, abs(shift[0])*1.5))
            ax.set_title(f'Rigid Shift for Frame {frame_idx}')
            ax.set_xlabel('X Shift (pixels)')
            ax.set_ylabel('Y Shift (pixels)')
            ax.grid(True)
            ax.set_aspect('equal')
            
            return fig
        else:
            # No shifts to display
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No shifts to display', 
                    ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
    
    def visualize_correction(self, original, corrected, frame_idx):
        """
        Visualize correction results for a specific frame.
        
        Parameters
        ----------
        original : numpy.ndarray
            Original video
        corrected : numpy.ndarray
            Corrected video
        frame_idx : int
            Frame index to visualize
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with correction visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original frame
        axes[0].imshow(original[frame_idx], cmap='gray')
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Plot corrected frame
        axes[1].imshow(corrected[frame_idx], cmap='gray')
        axes[1].set_title('Corrected Frame')
        axes[1].axis('off')
        
        # Plot difference
        diff = original[frame_idx] - corrected[frame_idx]
        vmax = max(abs(diff.min()), abs(diff.max()))
        im = axes[2].imshow(diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        return fig

def correct_motion(video_data, output_dir=None, template=None, 
                  rigid_first=True, max_shifts=(20, 20), 
                  grid_size=(6, 6), patch_size=(48, 48),
                  iterations=2, n_processes=8, batch_size=100,
                  save_results=True, visualize=True):
    """
    Apply optimized non-rigid motion correction to a video.
    
    Parameters
    ----------
    video_data : numpy.ndarray
        Input video with shape (t, h, w)
    output_dir : str, optional
        Directory to save results, by default None
    template : numpy.ndarray, optional
        Custom template for registration, by default None
    rigid_first : bool, optional
        Whether to apply rigid correction before non-rigid, by default True
    max_shifts : tuple, optional
        Maximum allowed shifts in pixels (y, x), by default (20, 20)
    grid_size : tuple, optional
        Grid size for non-rigid registration (y, x), by default (6, 6)
    patch_size : tuple, optional
        Size of patches for non-rigid registration (y, x), by default (48, 48)
    iterations : int, optional
        Number of iterations for refinement, by default 2
    n_processes : int, optional
        Number of processes for parallel computation, by default 8
    batch_size : int, optional
        Number of frames to process in one batch, by default 100
    save_results : bool, optional
        Whether to save results, by default True
    visualize : bool, optional
        Whether to generate visualizations, by default True
        
    Returns
    -------
    tuple
        (corrected_video, shifts_dict, template)
    """
    # Create motion correction object
    mc = OptimizedNonRigidCorrection(
        max_shifts=max_shifts,
        grid_size=grid_size,
        patch_size=patch_size,
        iterations=iterations,
        n_processes=n_processes,
        batch_size=batch_size,
        verbose=True
    )
    
    # Apply correction
    print(f"Starting optimized non-rigid motion correction...")
    print(f"Parameters:")
    print(f" - Rigid first: {rigid_first}")
    print(f" - Grid size: {grid_size}")
    print(f" - Patch size: {patch_size}")
    print(f" - Max shifts: {max_shifts} pixels")
    print(f" - Batch size: {batch_size} frames")
    print(f" - Parallel processes: {n_processes}")
    
    start_time = time.time()
    
    corrected_video, shifts_dict, template = mc.correct(
        video_data, template=template, rigid_first=rigid_first
    )
    
    elapsed_time = time.time() - start_time
    print(f"Motion correction completed in {elapsed_time:.2f} seconds "
          f"({video_data.shape[0]/elapsed_time:.2f} frames/sec)")
    
    # Save results if requested
    if save_results and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save corrected video as 16-bit TIFF
        tiff_path = os.path.join(output_dir, 'corrected_video.tif')
        # Convert to 16-bit (uint16)
        corrected_uint16 = (corrected_video * 65535).astype(np.uint16)
        imwrite(tiff_path, corrected_uint16, photometric='minisblack')
        
        # Generate visualizations
        if visualize:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Visualize shifts
            fig = mc.visualize_shifts(shifts_dict)
            fig.savefig(os.path.join(vis_dir, 'shifts_visualization.png'))
            plt.close(fig)
            
            # Visualize correction for sample frames
            n_frames = corrected_video.shape[0]
            sample_frames = np.linspace(0, n_frames-1, min(5, n_frames), dtype=int)
            
            for idx in sample_frames:
                fig = mc.visualize_correction(video_data, corrected_video, idx)
                fig.savefig(os.path.join(vis_dir, f'correction_frame_{idx}.png'))
                plt.close(fig)
    
    return corrected_video, shifts_dict, template