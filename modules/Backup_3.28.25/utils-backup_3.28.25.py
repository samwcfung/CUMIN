#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities Module
--------------
Utility functions for the fluorescence analysis pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def setup_logging(log_file=None, process_id=None):
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Path to save log file, or None for console only
    process_id : int, optional
        Process ID for parallel processing
        
    Returns
    -------
    logging.Logger
        Configured logger object
    """
    # Create logger
    if process_id is not None:
        logger = logging.getLogger(f"pipeline_process_{process_id}")
    else:
        logger = logging.getLogger("pipeline")
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_slice_data(slice_name, metrics_df, output_dir, logger):
    """
    Save slice data to Excel and JSON files.
    
    Parameters
    ----------
    slice_name : str
        Name of the slice
    metrics_df : pandas.DataFrame
        DataFrame containing metrics for each ROI
    output_dir : str
        Directory to save the data
    logger : logging.Logger
        Logger object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to Excel
    excel_path = os.path.join(output_dir, f"{slice_name}_metrics.xlsx")
    metrics_df.to_excel(excel_path, index=False)
    
    # Save metrics to JSON for easier programmatic access
    json_path = os.path.join(output_dir, f"{slice_name}_metrics.json")
    
    # Convert DataFrame to dict for JSON serialization
    metrics_dict = metrics_df.to_dict(orient='records')
    
    # Ensure all values are JSON serializable
    for roi_metrics in metrics_dict:
        for key, value in roi_metrics.items():
            # Convert numpy types to Python native types
            if isinstance(value, np.integer):
                roi_metrics[key] = int(value)
            elif isinstance(value, np.floating):
                roi_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                roi_metrics[key] = value.tolist()
    
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Saved slice data to {excel_path} and {json_path}")

def save_mouse_summary(mouse_id, slice_results, output_dir, logger):
    """
    Create a summary Excel file for a mouse containing data from all slices,
    with each slice as a separate row.
    
    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    slice_results : list
        List of slice result dictionaries
    output_dir : str
        Directory to save the summary
    logger : logging.Logger
        Logger object
    """
    logger.info(f"Creating summary for mouse {mouse_id}")
    
    # Create a new Excel writer
    summary_path = os.path.join(output_dir, f"{mouse_id}_summary.xlsx")
    writer = pd.ExcelWriter(summary_path, engine='xlsxwriter')
    
    # List to store summary data with one row per slice
    summary_data = []
    
    # Process each slice result
    for result in slice_results:
        slice_name = result["slice_name"]
        metrics_file = result["metrics_file"]
        metadata = result["metadata"]
        
        # Read metrics file
        try:
            metrics_df = pd.read_excel(metrics_file)
            
            # Create a more concise sheet name for Excel
            sheet_name = f"{metadata.get('slice_type', '')}{metadata.get('slice_number', '')}_" \
                         f"{metadata.get('condition', '')}"
            
            # Ensure sheet name is valid for Excel (max 31 chars, no special chars)
            sheet_name = sheet_name[:31].replace('/', '_').replace('\\', '_')
            if not sheet_name or sheet_name.isspace():
                sheet_name = slice_name[:31]
            
            # Add metrics as a sheet in the Excel file
            metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create summary row for this specific slice
            summary_row = {
                "Slice Name": slice_name,
                "Condition": metadata.get("condition", "unknown"),
                "Slice Type": metadata.get("slice_type", "unknown"),
                "Slice Number": metadata.get("slice_number", "1"),
                "Total ROIs": len(metrics_df),
                "Active ROIs": metrics_df["is_active"].sum(),
                "Active ROI %": (metrics_df["is_active"].sum() / len(metrics_df) * 100) if len(metrics_df) > 0 else 0,
                "Mean Peak Amplitude": metrics_df["peak_amplitude"].mean() if "peak_amplitude" in metrics_df.columns else 0,
                "Std Peak Amplitude": metrics_df["peak_amplitude"].std() if "peak_amplitude" in metrics_df.columns and len(metrics_df) > 1 else 0,
                "Mean Distance to Lamina": metrics_df["distance_to_lamina"].mean() if "distance_to_lamina" in metrics_df.columns else 0,
                "Mean Spontaneous Frequency": metrics_df["spont_peak_frequency"].mean() if "spont_peak_frequency" in metrics_df.columns else 0,
                "Mean Area Under Curve": metrics_df["peak_area_under_curve"].mean() if "peak_area_under_curve" in metrics_df.columns else 0
            }
            
            summary_data.append(summary_row)
            
        except Exception as e:
            logger.error(f"Error processing metrics file {metrics_file}: {str(e)}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by condition, slice type, and slice number
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Condition", "Slice Type", "Slice Number"])
    
    # Write summary sheet
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Format summary sheet
    workbook = writer.book
    summary_sheet = writer.sheets["Summary"]
    
    # Add percentage format
    pct_format = workbook.add_format({'num_format': '0.0%'})
    summary_sheet.set_column('F:F', None, pct_format)  # Apply to Active ROI % column
    
    # Close writer
    writer.close()
    
    logger.info(f"Saved mouse summary to {summary_path}")

def calculate_lamina_distance(roi_masks, image_shape):
    """
    Calculate distance from each ROI center to the top of the image (lamina border).
    
    Parameters
    ----------
    roi_masks : list
        List of ROI masks
    image_shape : tuple
        Tuple of (height, width) defining the image dimensions
        
    Returns
    -------
    list
        List of distances (in pixels) from ROI center to top of image
    """
    distances = []
    
    for mask in roi_masks:
        # Find ROI center
        y_indices, x_indices = np.where(mask)
        center_y = int(np.mean(y_indices))
        
        # Distance to top (lamina border)
        distance = center_y
        
        distances.append(distance)
    
    return distances

def save_requirements_txt(output_dir):
    """
    Generate a requirements.txt file with all dependencies.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the requirements file
    """
    requirements = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "h5py>=3.4.0",
        "tifffile>=2021.7.2",
        "opencv-python>=4.5.3",
        "seaborn>=0.11.2",
        "PyYAML>=6.0",
        "openpyxl>=3.0.9",
        "xlsxwriter>=3.0.2",
        "tqdm>=4.62.0"
    ]
    
    # Optional dependencies
    optional_requirements = [
        "# Optional dependencies",
        "cupy>=10.0.0  # For GPU acceleration",
        "ijroi>=0.2.2  # For advanced ROI handling"
    ]
    
    # Write requirements to file
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))
        f.write("\n\n")
        f.write("\n".join(optional_requirements))