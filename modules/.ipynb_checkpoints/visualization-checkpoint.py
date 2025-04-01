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
    df_f_traces,  # Renamed from fluorescence_data to df_f_traces
    roi_masks,
    metrics_df,
    flagged_rois,
    tif_path,
    output_dir,
    config,
    logger,
    metadata=None  # Add metadata parameter
):
    """
    Generate visualizations for fluorescence data analysis.
    
    Parameters
    ----------
    df_f_traces : numpy.ndarray
        dF/F traces (baseline normalized to 0) with shape (n_rois, n_frames)
    roi_masks : list
        List of ROI masks
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    flagged_rois : list
        List of ROI IDs with quality issues
    tif_path : str
        Path to the original .tif image
    output_dir : str
        Directory to save visualizations
    config : dict
        Visualization configuration
    logger : logging.Logger
        Logger object
    metadata : dict, optional
        Metadata dictionary with condition information
    """
    logger.info("Generating visualizations using dF/F traces")
    
    # Get condition from metadata if available
    condition = metadata.get("condition", "unknown") if metadata else "unknown"
    logger.info(f"Generating visualizations for condition: {condition}")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get the visualization sub-config first, providing an empty dict as default
    viz_config = config.get("visualization", {})
    # Then get plot_types from the sub-config
    plot_types = viz_config.get("plot_types", ["roi_map", "trace_overlay", "event_summary"])
    
    # Condition-specific visualization adjustments
    if condition == "0um":
        logger.info("Generating spontaneous activity visualizations for 0um condition")
        # For 0um, focus on spontaneous activity in the visualizations
        title_suffix = " (Spontaneous Activity)"
        if "trace_overlay" in plot_types:
            # Create spontaneous activity overlay
            create_trace_overlay(
                df_f_traces,  # FIXED: Using df_f_traces instead of fluorescence_data 
                metrics_df, 
                os.path.join(vis_dir, "spontaneous_trace_overlay.png"),
                title=f"Fluorescence Traces{title_suffix}",
                highlight_frame_range=None,  # No specific highlight region for spontaneous
                logger=logger
            )
    elif condition in ["10um", "25um"]:
        logger.info(f"Generating evoked activity visualizations for {condition} condition")
        # For 10um and 25um, focus on evoked activity in the visualizations
        title_suffix = f" (Evoked Activity - {condition})"
        if "trace_overlay" in plot_types:
            # Create evoked activity overlay with highlighted region
            create_trace_overlay(
                df_f_traces,  # FIXED: Using df_f_traces instead of fluorescence_data
                metrics_df, 
                os.path.join(vis_dir, "evoked_trace_overlay.png"),
                title=f"Fluorescence Traces{title_suffix}",
                highlight_frame_range = config["analysis"]["condition_specific"][condition]["analysis_frames"],  # Highlight the evoked activity region
                logger=logger
            )
    else:
        # Default visualizations
        title_suffix = ""
        if "trace_overlay" in plot_types:
            create_trace_overlay(
                df_f_traces,  # FIXED: Using df_f_traces instead of fluorescence_data
                metrics_df, 
                os.path.join(vis_dir, "trace_overlay.png"),
                logger=logger
            )
    
    # Generate additional plots based on the plot_types configuration
    for plot_type in plot_types:
        if plot_type == "roi_map":
            create_roi_map(
                roi_masks,
                metrics_df,
                tif_path,
                os.path.join(vis_dir, f"roi_map{title_suffix.replace(' ', '_')}.png"),
                colormap=config.get("roi_color_map", "viridis"),
                logger=logger
            )
        elif plot_type == "event_summary":
            # For event summary, adjust based on condition
            if condition == "0um":
                # For 0um, focus on spontaneous activity metrics
                create_event_summary(
                    metrics_df,
                    os.path.join(vis_dir, f"spontaneous_event_summary.png"),
                    title=f"Spontaneous Activity Summary{title_suffix}",
                    x_metric="spont_peak_frequency",
                    y_metric="spont_avg_peak_amplitude",
                    color_metric="is_active",
                    logger=logger
                )
            elif condition in ["10um", "25um"]:
                # For 10um and 25um, focus on evoked activity metrics
                create_event_summary(
                    metrics_df,
                    os.path.join(vis_dir, f"evoked_event_summary.png"),
                    title=f"Evoked Activity Summary{title_suffix}",
                    x_metric="peak_amplitude",
                    y_metric="peak_area_under_curve",
                    color_metric="is_active",
                    logger=logger
                )
            else:
                # Default behavior
                create_event_summary(
                    metrics_df,
                    os.path.join(vis_dir, f"event_summary.png"),
                    logger=logger
                )
        elif plot_type == "quality_metrics":
            create_quality_metrics_plot(
                metrics_df,
                flagged_rois,
                os.path.join(vis_dir, f"quality_metrics{title_suffix.replace(' ', '_')}.png"),
                logger=logger
            )
    
    # Individual ROI plots if enabled
    if config.get("save_individual_plots", True):
        roi_plots_dir = os.path.join(vis_dir, "individual_rois")
        os.makedirs(roi_plots_dir, exist_ok=True)
        
        for i in range(len(roi_masks)):
            roi_id = i + 1
            is_active = metrics_df.loc[metrics_df['roi_id'] == roi_id, 'is_active'].values[0]
            
            # Create different plot types based on condition
            if condition == "0um":
                create_individual_roi_plot(
                    df_f_traces[i],  # FIXED: Using df_f_traces instead of fluorescence_data
                    roi_id,
                    is_active,
                    os.path.join(roi_plots_dir, f"roi_{roi_id:03d}.png"),
                    title=f"ROI {roi_id} - Spontaneous Activity",
                    highlight_range=None,  # No highlight for spontaneous
                    logger=logger
                )
            elif condition in ["10um", "25um"]:
                create_individual_roi_plot(
                    df_f_traces[i],  # FIXED: Using df_f_traces instead of fluorescence_data
                    roi_id,
                    is_active,
                    os.path.join(roi_plots_dir, f"roi_{roi_id:03d}.png"),
                    title=f"ROI {roi_id} - Evoked Activity",
                    highlight_range=[100, 580],  # Highlight evoked activity region
                    logger=logger
                )
            else:
                create_individual_roi_plot(
                    df_f_traces[i],  # FIXED: Using df_f_traces instead of fluorescence_data
                    roi_id,
                    is_active,
                    os.path.join(roi_plots_dir, f"roi_{roi_id:03d}.png"),
                    logger=logger
                )
    
    logger.info(f"Generated visualizations in {vis_dir}")


def create_individual_roi_plot(
    df_f_trace,  # Renamed from trace to df_f_trace
    roi_id,
    is_active,
    output_path,
    title=None,
    highlight_range=None,
    logger=None
):
    """
    Create a plot for an individual ROI.
    
    Parameters
    ----------
    df_f_trace : numpy.ndarray
        dF/F trace for the ROI (baseline normalized to 0)
    roi_id : int
        ROI identifier
    is_active : bool
        Whether the ROI is active
    output_path : str
        Path to save the plot
    title : str, optional
        Plot title
    highlight_range : list, optional
        Range of frames to highlight [start, end]
    logger : logging.Logger, optional
        Logger object
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot the trace
        x = np.arange(len(df_f_trace))
        plt.plot(x, df_f_trace, 'b-', linewidth=1.5)
        
        # Highlight specific range if provided
        if highlight_range is not None:
            start, end = highlight_range
            start = max(0, start)
            end = min(len(df_f_trace)-1, end)
            
            plt.axvspan(start, end, color='lightgray', alpha=0.3, label='Analysis Window')
        
        # Set title
        if title is None:
            title = f"ROI {roi_id}"
        
        # Add active/inactive to title
        if is_active:
            plt.title(f"{title} (Active)", fontsize=14)
        else:
            plt.title(f"{title} (Inactive)", fontsize=14)
        
        plt.xlabel("Frame", fontsize=12)
        plt.ylabel("dF/F", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # If there's a highlight, add a legend
        if highlight_range is not None:
            plt.legend()
        
        # Add some padding to the x-axis
        plt.xlim(-len(df_f_trace)*0.02, len(df_f_trace)*1.02)
        
        # Add a zero line to reference baseline
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        if logger:
            logger.info(f"Saved individual ROI plot to {output_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error creating individual ROI plot: {str(e)}")


def create_trace_overlay(
    df_f_traces,  # Renamed from fluorescence_data to df_f_traces
    metrics_df,
    output_path,
    title="Fluorescence Traces Overlay",
    highlight_frame_range=None,
    logger=None
):
    """
    Create an overlay of all ROI traces.
    
    Parameters
    ----------
    df_f_traces : numpy.ndarray
        dF/F traces with shape (n_rois, n_frames), baseline normalized to 0
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    output_path : str
        Path to save the plot
    title : str, optional
        Plot title
    highlight_frame_range : list, optional
        Range of frames to highlight [start, end]
    logger : logging.Logger, optional
        Logger object
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Get active ROIs
        active_mask = metrics_df['is_active'].values
        
        # Create x values for the plot
        x = np.arange(df_f_traces.shape[1])
        
        # Plot inactive ROIs first (in gray)
        for i in range(df_f_traces.shape[0]):
            if not active_mask[i]:
                plt.plot(x, df_f_traces[i], color='gray', linewidth=0.5, alpha=0.3)
        
        # Plot active ROIs on top (in color)
        for i in range(df_f_traces.shape[0]):
            if active_mask[i]:
                plt.plot(x, df_f_traces[i], linewidth=1, alpha=0.7)
        
        # Highlight specific range if provided
        if highlight_frame_range is not None:
            start, end = highlight_frame_range
            start = max(0, start)
            end = min(df_f_traces.shape[1]-1, end)
            
            plt.axvspan(start, end, color='lightgray', alpha=0.3, label='Analysis Window')
            plt.legend()
        
        plt.title(title, fontsize=14)
        plt.xlabel("Frame", fontsize=12)
        plt.ylabel("dF/F", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add a zero line to reference baseline
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add some padding to the x-axis
        plt.xlim(-df_f_traces.shape[1]*0.02, df_f_traces.shape[1]*1.02)
        
        # Create legend indicating number of active vs inactive ROIs
        n_active = np.sum(active_mask)
        n_inactive = df_f_traces.shape[0] - n_active
        
        plt.figtext(0.02, 0.02, f"Active ROIs: {n_active}\nInactive ROIs: {n_inactive}", 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        if logger:
            logger.info(f"Saved trace overlay plot to {output_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error creating trace overlay plot: {str(e)}")


def create_event_summary(
    metrics_df,
    output_path,
    title="Event Summary",
    x_metric="peak_amplitude",
    y_metric="peak_area_under_curve",
    color_metric="is_active",
    logger=None
):
    """
    Create a summary plot of event metrics.
    
    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    output_path : str
        Path to save the plot
    title : str, optional
        Plot title
    x_metric : str, optional
        Metric to plot on x-axis
    y_metric : str, optional
        Metric to plot on y-axis
    color_metric : str, optional
        Metric to use for coloring points
    logger : logging.Logger, optional
        Logger object
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        if color_metric == "is_active":
            # Use active/inactive as color categories
            colors = metrics_df['is_active'].map({True: 'green', False: 'gray'})
            
            plt.scatter(
                metrics_df[x_metric],
                metrics_df[y_metric],
                c=colors,
                alpha=0.7,
                s=50,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add a legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Active'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Inactive')
            ]
            plt.legend(handles=legend_elements)
        else:
            # Use a continuous colormap for other metrics
            scatter = plt.scatter(
                metrics_df[x_metric],
                metrics_df[y_metric],
                c=metrics_df[color_metric],
                cmap='viridis',
                alpha=0.7,
                s=50,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add a colorbar
            plt.colorbar(scatter, label=color_metric)
        
        plt.title(title, fontsize=14)
        plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add a text box with summary statistics
        n_active = metrics_df['is_active'].sum()
        n_total = len(metrics_df)
        percent_active = (n_active / n_total) * 100 if n_total > 0 else 0
        
        stats_text = (
            f"Total ROIs: {n_total}\n"
            f"Active ROIs: {n_active} ({percent_active:.1f}%)\n"
            f"Mean {x_metric}: {metrics_df[x_metric].mean():.3f}\n"
            f"Mean {y_metric}: {metrics_df[y_metric].mean():.3f}"
        )
        
        plt.figtext(0.02, 0.02, stats_text, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        if logger:
            logger.info(f"Saved event summary plot to {output_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error creating event summary plot: {str(e)}")


def create_roi_map(
    roi_masks,
    metrics_df,
    tif_path,
    output_path,
    colormap='viridis',
    logger=None
):
    """
    Create a map of ROIs colored by activity.
    
    Parameters
    ----------
    roi_masks : list
        List of ROI masks
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    tif_path : str
        Path to the original .tif image
    output_path : str
        Path to save the plot
    colormap : str, optional
        Colormap to use
    logger : logging.Logger, optional
        Logger object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import tifffile
        
        # Load the first frame of the TIF file
        with tifffile.TiffFile(tif_path) as tif:
            if len(tif.pages) > 0:
                background = tif.pages[0].asarray()
            else:
                # Create blank background
                if len(roi_masks) > 0:
                    background = np.zeros_like(roi_masks[0], dtype=np.uint16)
                else:
                    background = np.zeros((512, 512), dtype=np.uint16)
        
        plt.figure(figsize=(10, 10))
        
        # Display background in grayscale
        plt.imshow(background, cmap='gray')
        
        # Create colored ROI overlay
        roi_overlay = np.zeros((*background.shape, 4))  # RGBA
        
        # Get active status for each ROI
        active_status = metrics_df['is_active'].values
        
        for i, mask in enumerate(roi_masks):
            roi_id = i + 1
            
            # Find ROI in metrics_df
            is_active = active_status[i] if i < len(active_status) else False
            
            # Color based on active status
            if is_active:
                color = np.array([0, 1, 0, 0.5])  # Green, semi-transparent
            else:
                color = np.array([1, 0, 0, 0.3])  # Red, more transparent
            
            # Apply color to mask
            for c in range(4):
                roi_overlay[:, :, c] = np.where(mask, color[c], roi_overlay[:, :, c])
            
            # Add ROI label
            y, x = np.where(mask)
            if len(y) > 0 and len(x) > 0:
                center_y = int(np.mean(y))
                center_x = int(np.mean(x))
                plt.text(center_x, center_y, str(roi_id), color='white', fontsize=8,
                        ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=0))
        
        # Overlay colored ROIs
        plt.imshow(roi_overlay)
        
        # Create legend
        active_patch = mpatches.Patch(color='green', alpha=0.5, label='Active ROI')
        inactive_patch = mpatches.Patch(color='red', alpha=0.3, label='Inactive ROI')
        plt.legend(handles=[active_patch, inactive_patch], loc='upper right')
        
        # Add title with ROI counts
        n_active = np.sum(active_status)
        plt.title(f"ROI Map ({n_active}/{len(roi_masks)} Active)", fontsize=14)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        
        if logger:
            logger.info(f"Saved ROI map to {output_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error creating ROI map: {str(e)}")


def create_quality_metrics_plot(
    metrics_df,
    flagged_rois,
    output_path,
    logger=None
):
    """
    Create a plot of quality metrics for ROIs.
    
    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    flagged_rois : list
        List of ROI IDs with quality issues
    output_path : str
        Path to save the plot
    logger : logging.Logger, optional
        Logger object
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert flagged_rois to a set of ROI IDs for easier lookup
        flagged_roi_ids = {flag['roi_id'] for flag in flagged_rois}
        
        # Add a column to metrics_df indicating if ROI is flagged
        metrics_df['is_flagged'] = metrics_df['roi_id'].isin(flagged_roi_ids)
        
        plt.figure(figsize=(12, 10))
        
        # Create a 2x2 grid of quality metric plots
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=metrics_df,
            x='baseline_fluorescence',
            y='std_df_f',
            hue='is_flagged',
            palette={True: 'red', False: 'blue'},
            s=50
        )
        plt.title("Signal Variability vs Baseline")
        plt.xlabel("Baseline Fluorescence")
        plt.ylabel("Signal Std. Dev.")
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            data=metrics_df,
            x='roi_id',
            y='mean_df_f',
            hue='is_flagged',
            palette={True: 'red', False: 'blue'},
            s=50
        )
        plt.title("Mean dF/F by ROI")
        plt.xlabel("ROI ID")
        plt.ylabel("Mean dF/F")
        
        plt.subplot(2, 2, 3)
        sns.histplot(
            data=metrics_df,
            x='baseline_fluorescence',
            hue='is_flagged',
            palette={True: 'red', False: 'blue'},
            element="step",
            common_norm=False,
            alpha=0.5
        )
        plt.title("Baseline Fluorescence Distribution")
        plt.xlabel("Baseline Fluorescence")
        plt.ylabel("Count")
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(
            data=metrics_df,
            x='distance_to_lamina',
            y='peak_amplitude',
            hue='is_flagged',
            palette={True: 'red', False: 'blue'},
            s=50
        )
        plt.title("Peak Amplitude vs Distance")
        plt.xlabel("Distance to Lamina")
        plt.ylabel("Peak Amplitude")
        
        # Add a summary text
        plt.figtext(0.5, 0.01, 
                  f"Total ROIs: {len(metrics_df)}, Flagged ROIs: {len(flagged_roi_ids)} ({len(flagged_roi_ids)/len(metrics_df)*100:.1f}%)",
                  ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        if logger:
            logger.info(f"Saved quality metrics plot to {output_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error creating quality metrics plot: {str(e)}")