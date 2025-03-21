#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module
------------------
Generate visualizations for verification and QC.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
import tifffile
import cv2

def generate_visualizations(
    fluorescence_data, 
    roi_masks, 
    metrics_df, 
    flagged_rois, 
    original_image_path, 
    output_dir, 
    vis_config, 
    logger
):
    """
    Generate visualizations for verification and QC.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    flagged_rois : list
        List of flagged ROI IDs with issues
    original_image_path : str
        Path to the original .tif image
    output_dir : str
        Directory to save the visualizations
    vis_config : dict
        Visualization configuration parameters
    logger : logging.Logger
        Logger object
    """
    start_time = time.time()
    logger.info("Generating visualizations")
    
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract visualization settings
    baseline_frames = vis_config.get("baseline_frames", [0, 200])
    analysis_frames = vis_config.get("analysis_frames", [233, 580])
    
    # 1. Generate fluorescence heatmap
    generate_fluorescence_heatmap(
        fluorescence_data,
        baseline_frames,
        analysis_frames,
        os.path.join(vis_dir, "fluorescence_heatmap.png"),
        logger
    )
    
    # 2. Plot traces for randomly selected ROIs
    generate_random_roi_traces(
        fluorescence_data,
        baseline_frames,
        analysis_frames,
        os.path.join(vis_dir, "random_roi_traces.png"),
        logger
    )
    
    # 3. Overlay ROI masks on original image
    generate_roi_overlay(
        roi_masks,
        original_image_path,
        os.path.join(vis_dir, "roi_overlay.png"),
        flagged_rois,
        logger
    )
    
    # 4. Generate time-series overlays comparing spontaneous vs. evoked responses
    generate_response_comparison(
        fluorescence_data,
        metrics_df,
        baseline_frames,
        analysis_frames,
        os.path.join(vis_dir, "response_comparison.png"),
        logger
    )
    
    # 5. Generate QC summary report
    generate_qc_report(
        flagged_rois,
        fluorescence_data,
        os.path.join(vis_dir, "qc_report.png"),
        logger
    )
    
    logger.info(f"Visualization generation completed in {time.time() - start_time:.2f} seconds")

def generate_fluorescence_heatmap(
    fluorescence_data, 
    baseline_frames, 
    analysis_frames, 
    output_path, 
    logger
):
    """
    Generate a heatmap of fluorescence intensities across ROIs.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    baseline_frames : list
        [start, end] frames for baseline
    analysis_frames : list
        [start, end] frames for analysis
    output_path : str
        Path to save the visualization
    logger : logging.Logger
        Logger object
    """
    n_rois, n_frames = fluorescence_data.shape
    
    # Check if we have data to plot
    if n_rois == 0 or n_frames == 0:
        logger.warning("No data to plot for heatmap visualization")
        # Create a dummy figure with a message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No fluorescence data available for heatmap", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    # Normalize each ROI by its baseline
    normalized_data = np.zeros_like(fluorescence_data)
    
    for i in range(n_rois):
        # Calculate baseline window, ensuring it's valid
        baseline_start = max(0, min(baseline_frames[0], n_frames-1))
        baseline_end = max(baseline_start, min(baseline_frames[1], n_frames-1))
        
        if baseline_end >= baseline_start:
            baseline = np.mean(fluorescence_data[i, baseline_start:baseline_end+1])
            # Normalize, avoiding division by zero
            if baseline != 0:
                normalized_data[i] = (fluorescence_data[i] - baseline) / baseline
            else:
                normalized_data[i] = fluorescence_data[i]
        else:
            # If baseline window is invalid, just use raw data
            normalized_data[i] = fluorescence_data[i]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap with seaborn if available, fall back to matplotlib
    try:
        import seaborn as sns
        # This creates the mappable object for the colorbar
        heatmap = sns.heatmap(
            normalized_data, 
            cmap='viridis', 
            vmin=-0.2, 
            vmax=0.5, 
            xticklabels=50, 
            yticklabels=5
        )
        # No need for separate colorbar - seaborn heatmap includes one
    except (ImportError, Exception):
        # Fall back to matplotlib imshow
        heatmap = plt.imshow(
            normalized_data, 
            aspect='auto', 
            cmap='viridis',
            vmin=-0.2, 
            vmax=0.5
        )
        plt.colorbar(heatmap, label='dF/F')
    
    # Add vertical lines for baseline and analysis windows
    # Make sure we don't exceed array bounds
    for frame in [min(baseline_frames[0], n_frames-1), 
                  min(baseline_frames[1], n_frames-1),
                  min(analysis_frames[0], n_frames-1), 
                  min(analysis_frames[1], n_frames-1)]:
        if 0 <= frame < n_frames:
            # Use different colors for baseline and analysis
            color = 'red' if frame in baseline_frames else 'blue'
            plt.axvline(x=frame, color=color, linestyle='--', linewidth=1)
    
    # Add labels
    plt.xlabel('Frame')
    plt.ylabel('ROI')
    plt.title('Fluorescence Intensity Heatmap (dF/F)')
    
    # Save figure
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300)
    except Exception as e:
        logger.warning(f"Error saving heatmap figure: {str(e)}")
    plt.close()
    
    logger.info(f"Saved fluorescence heatmap to {output_path}")

def generate_random_roi_traces(
    fluorescence_data, 
    baseline_frames, 
    analysis_frames, 
    output_path, 
    logger
):
    """
    Generate plots of traces for randomly selected ROIs.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    baseline_frames : list
        [start, end] frames for baseline
    analysis_frames : list
        [start, end] frames for analysis
    output_path : str
        Path to save the visualization
    logger : logging.Logger
        Logger object
    """
    n_rois, n_frames = fluorescence_data.shape
    
    # Select 3 random ROIs (or fewer if not enough)
    n_random = min(3, n_rois)
    random_indices = random.sample(range(n_rois), n_random)
    
    # Create figure
    fig, axes = plt.subplots(n_random, 1, figsize=(10, 3*n_random), sharex=True)
    
    # Make sure axes is always a list
    if n_random == 1:
        axes = [axes]
    
    # Plot each selected ROI
    for i, roi_idx in enumerate(random_indices):
        trace = fluorescence_data[roi_idx]
        
        # Calculate baseline
        baseline = np.mean(trace[baseline_frames[0]:baseline_frames[1]+1])
        
        # Calculate dF/F
        df_f = (trace - baseline) / baseline if baseline != 0 else trace
        
        # Plot trace
        axes[i].plot(df_f, linewidth=1)
        
        # Add vertical lines for baseline and analysis windows
        axes[i].axvspan(baseline_frames[0], baseline_frames[1], alpha=0.2, color='gray')
        axes[i].axvspan(analysis_frames[0], analysis_frames[1], alpha=0.2, color='blue')
        
        # Add labels
        axes[i].set_ylabel('dF/F')
        axes[i].set_title(f'ROI {roi_idx + 1}')
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add common x-label
    axes[-1].set_xlabel('Frame')
    
    # Add legend
    fig.legend(['Fluorescence', 'Baseline', 'Analysis'], loc='upper center', ncol=3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved random ROI traces to {output_path}")

def generate_roi_overlay(
    roi_masks, 
    original_image_path, 
    output_path, 
    flagged_rois, 
    logger
):
    """
    Generate an overlay of ROI masks on the original image.
    
    Parameters
    ----------
    roi_masks : list
        List of ROI masks
    original_image_path : str
        Path to the original .tif image
    output_path : str
        Path to save the visualization
    flagged_rois : list
        List of flagged ROI IDs with issues
    logger : logging.Logger
        Logger object
    """
    # Load the first frame of the original image
    try:
        with tifffile.TiffFile(original_image_path) as tif:
            if len(tif.pages) > 0:
                original_image = tif.pages[0].asarray()
            else:
                raise ValueError("No frames found in the original .tif file")
    except Exception as e:
        logger.error(f"Error loading original image: {str(e)}")
        return
    
    # Normalize to 8-bit for visualization
    original_image = cv2.normalize(
        original_image, 
        None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )
    
    # Convert to RGB
    overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Get list of flagged ROI IDs
    flagged_ids = [flag['roi_id'] for flag in flagged_rois]
    
    # Overlay each ROI mask
    for i, mask in enumerate(roi_masks):
        roi_id = i + 1
        
        # Choose color based on whether ROI is flagged
        if roi_id in flagged_ids:
            color = (255, 0, 0)  # Red for flagged ROIs
        else:
            color = (0, 255, 0)  # Green for normal ROIs
        
        # Create a contour of the mask
        y_indices, x_indices = np.where(mask)
        points = np.column_stack((x_indices, y_indices))
        
        if len(points) > 0:
            # Find contour of ROI
            try:
                hull = cv2.convexHull(points)
                cv2.drawContours(overlay_image, [hull], 0, color, 2)
                
                # Find centroid
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                
                # Add ROI ID
                cv2.putText(
                    overlay_image,
                    str(roi_id),
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
            except Exception as e:
                logger.warning(f"Error drawing contour for ROI {roi_id}: {str(e)}")
    
    # Save overlay image
    cv2.imwrite(output_path, overlay_image)
    
    logger.info(f"Saved ROI overlay to {output_path}")

def generate_response_comparison(
    fluorescence_data, 
    metrics_df, 
    baseline_frames, 
    analysis_frames, 
    output_path, 
    logger
):
    """
    Generate time-series overlays comparing spontaneous vs. evoked responses.
    
    Parameters
    ----------
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    baseline_frames : list
        [start, end] frames for baseline
    analysis_frames : list
        [start, end] frames for analysis
    output_path : str
        Path to save the visualization
    logger : logging.Logger
        Logger object
    """
    n_rois, n_frames = fluorescence_data.shape
    
    # Select active ROIs (those with significant evoked response)
    active_indices = metrics_df[metrics_df['is_active']]['roi_id'].values - 1  # Adjust for 0-indexing
    
    # If fewer than 2 active ROIs, select 2 random ROIs
    if len(active_indices) < 2:
        active_indices = np.random.choice(n_rois, min(2, n_rois), replace=False)
    
    # Select up to 4 active ROIs for display
    display_indices = active_indices[:min(4, len(active_indices))]
    
    # Create figure
    fig, axes = plt.subplots(len(display_indices), 1, figsize=(10, 3*len(display_indices)), sharex=True)
    
    # Make sure axes is always a list
    if len(display_indices) == 1:
        axes = [axes]
    
    # Plot each selected ROI
    for i, roi_idx in enumerate(display_indices):
        trace = fluorescence_data[int(roi_idx)]
        
        # Calculate baseline
        baseline = np.mean(trace[baseline_frames[0]:baseline_frames[1]+1])
        
        # Calculate dF/F
        df_f = (trace - baseline) / baseline if baseline != 0 else trace
        
        # Separate spontaneous and evoked periods
        spont_period = df_f[baseline_frames[0]:baseline_frames[1]+1]
        evoked_period = df_f[analysis_frames[0]:analysis_frames[1]+1]
        
        # Plot full trace in gray
        axes[i].plot(np.arange(n_frames), df_f, color='gray', alpha=0.5, linewidth=1)
        
        # Plot spontaneous period in blue
        axes[i].plot(
            np.arange(baseline_frames[0], baseline_frames[1]+1),
            spont_period,
            color='blue',
            linewidth=1.5
        )
        
        # Plot evoked period in red
        axes[i].plot(
            np.arange(analysis_frames[0], analysis_frames[1]+1),
            evoked_period,
            color='red',
            linewidth=1.5
        )
        
        # Add shaded regions for baseline and analysis windows
        axes[i].axvspan(baseline_frames[0], baseline_frames[1], alpha=0.1, color='blue')
        axes[i].axvspan(analysis_frames[0], analysis_frames[1], alpha=0.1, color='red')
        
        # Add labels
        axes[i].set_ylabel('dF/F')
        axes[i].set_title(f'ROI {int(roi_idx) + 1}')
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add common x-label
    axes[-1].set_xlabel('Frame')
    
    # Add legend
    fig.legend(
        ['Full Trace', 'Spontaneous Activity', 'Evoked Response', 'Baseline Period', 'Analysis Period'],
        loc='upper center',
        ncol=3
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved response comparison to {output_path}")

def generate_qc_report(
    flagged_rois, 
    fluorescence_data, 
    output_path, 
    logger
):
    """
    Generate a QC report showing traces of flagged ROIs.
    
    Parameters
    ----------
    flagged_rois : list
        List of flagged ROI IDs with issues
    fluorescence_data : numpy.ndarray
        Background-corrected fluorescence traces with shape (n_rois, n_frames)
    output_path : str
        Path to save the visualization
    logger : logging.Logger
        Logger object
    """
    if not flagged_rois:
        logger.info("No flagged ROIs to report")
        
        # Create a simple figure stating no issues
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No QC issues detected", fontsize=16, ha='center')
        plt.axis('off')
        plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    # Group flagged ROIs by issue type
    issue_types = sorted(set(flag['issue'] for flag in flagged_rois))
    
    # Create figure with subplots for each issue type
    fig, axes = plt.subplots(
        len(issue_types), 
        1, 
        figsize=(10, 4 * len(issue_types)), 
        sharex=True
    )
    
    # Make sure axes is always a list
    if len(issue_types) == 1:
        axes = [axes]
    
    # Plot each issue type
    for i, issue in enumerate(issue_types):
        # Get ROIs with this issue
        issue_rois = [flag for flag in flagged_rois if flag['issue'] == issue]
        
        # Plot traces for these ROIs
        for flag in issue_rois:
            roi_idx = flag['roi_id'] - 1  # Adjust for 0-indexing
            
            if roi_idx < fluorescence_data.shape[0]:
                trace = fluorescence_data[roi_idx]
                
                # Calculate baseline
                baseline = np.mean(trace[:20])  # Use first 20 frames as baseline
                
                # Calculate dF/F
                df_f = (trace - baseline) / baseline if baseline != 0 else trace
                
                # Plot trace
                axes[i].plot(df_f, alpha=0.7, linewidth=1, label=f"ROI {flag['roi_id']}")
                
                # Mark issue location if available
                if 'frame' in flag:
                    axes[i].axvline(x=flag['frame'], color='red', linestyle='--', alpha=0.5)
        
        # Add labels
        axes[i].set_title(f"Issue: {issue.replace('_', ' ').title()}")
        axes[i].set_ylabel('dF/F')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        if len(issue_rois) <= 10:  # Only show legend if not too many ROIs
            axes[i].legend(loc='upper right')
    
    # Add common x-label
    axes[-1].set_xlabel('Frame')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved QC report to {output_path}")