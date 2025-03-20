#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluorescence Extraction and Analysis Pipeline
--------------------------------------------
This script implements a modular pipeline for processing fluorescence imaging data
with parallel processing, error handling, and comprehensive logging.
"""

import os
import sys
import yaml
import logging
import argparse
import time
import random
import h5py
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Pipeline modules
from modules.file_matcher import match_tif_and_roi_files
from modules.preprocessing import correct_photobleaching
from modules.roi_processing import extract_roi_fluorescence, subtract_background
from modules.analysis import analyze_fluorescence, perform_qc_checks
from modules.visualization import generate_visualizations
from modules.utils import setup_logging, save_slice_data, save_mouse_summary
from modules.roi_processing import extract_rois_from_zip, save_masks_for_cnmf, extract_roi_fluorescence_with_cnmf

# Import advanced analysis module if available
try:
    from modules.advanced_analysis import run_advanced_analysis
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fluorescence Analysis Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .tif and .zip files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for pipeline outputs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--mode", type=str, choices=["all", "preprocess", "extract", "analyze"], 
                        default="all", help="Pipeline mode")
    parser.add_argument("--max_workers", type=int, default=None, 
                        help="Maximum number of parallel workers (default: number of CPUs)")
    parser.add_argument("--disable_advanced", action="store_true", 
                        help="Disable advanced analysis regardless of config setting")
    return parser.parse_args()

def load_config(config_path, args=None):
    """Load configuration from YAML file and apply command line overrides."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        
        # Apply command line overrides if provided
        if args and args.disable_advanced:
            # Disable advanced analysis if requested via command line
            if "advanced_analysis" in config:
                config["advanced_analysis"]["enabled"] = False
                logging.info("Advanced analysis disabled via command line argument")
        
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

def process_file_pair(tif_path, roi_path, output_dir, config, mode, process_id):
    """Process a matched .tif and .zip ROI file pair."""
    start_time = time.time()
    slice_name = Path(tif_path).stem
    
    # Setup slice-specific logging
    log_file = os.path.join(output_dir, 'logs', f"{slice_name}_process_{process_id}.log")
    slice_logger = setup_logging(log_file, process_id)
    slice_logger.info(f"Processing slice: {slice_name}")
    
    # Create output directories for this slice
    slice_output_dir = os.path.join(output_dir, slice_name)
    os.makedirs(slice_output_dir, exist_ok=True)
    
    try:
        # Extract metadata from filename
        metadata = extract_metadata_from_filename(slice_name)
        slice_logger.info(f"Extracted metadata: {metadata}")
        
        # Preprocessing
        if mode in ["all", "preprocess"]:
            slice_logger.info("Starting preprocessing")
            
            # Check if this is a CNMF/CNMF-E preprocessing method
            is_cnmf_method = config["preprocessing"].get("correction_method", "") in ["cnmf", "cnmf_e"]
            
            # If using CNMF-E, prepare masks from ROIs first
            if is_cnmf_method and config["preprocessing"].get("use_roi_masks", True):
                # Extract ROIs to use for CNMF-E initialization
                slice_logger.info("Extracting ROI masks for CNMF/CNMF-E initialization")
                from modules.roi_processing import extract_rois_from_zip, save_masks_for_cnmf
                
                # Get image shape from the first TIFF page
                import tifffile
                with tifffile.TiffFile(tif_path) as tif:
                    if len(tif.pages) > 0:
                        image_shape = (tif.pages[0].shape[0], tif.pages[0].shape[1])
                    else:
                        raise ValueError(f"Could not determine image shape from {tif_path}")
                
                # Extract ROIs and save them for CNMF-E
                roi_masks, _ = extract_rois_from_zip(roi_path, image_shape, slice_logger)
                if roi_masks:
                    masks_path = save_masks_for_cnmf(roi_masks, slice_output_dir, slice_logger)
                    # Update config to use these masks
                    config["preprocessing"]["masks_path"] = masks_path
                else:
                    slice_logger.warning("No valid ROIs found for CNMF/CNMF-E initialization")
            
            # Run photobleaching correction
            corrected_data, image_shape = correct_photobleaching(
                tif_path, 
                os.path.join(slice_output_dir, f"{slice_name}_corrected.h5"),
                config["preprocessing"],
                slice_logger
            )
            
            # Load CNMF components if they were generated
            cnmf_components = None
            if is_cnmf_method:
                components_path = os.path.join(slice_output_dir, f"{slice_name}_cnmf_components.h5")
                if os.path.exists(components_path):
                    slice_logger.info(f"Loading CNMF components from {components_path}")
                    try:
                        with h5py.File(components_path, 'r') as f:
                            if 'cnmf_components' in f:
                                # Load components into dictionary
                                cnmf_components = {}
                                for key in f['cnmf_components'].keys():
                                    cnmf_components[key] = f['cnmf_components'][key][:]
                                slice_logger.info(f"Loaded CNMF components: {list(cnmf_components.keys())}")
                    except Exception as e:
                        slice_logger.warning(f"Failed to load CNMF components: {str(e)}")
            
            preprocessing_time = time.time() - start_time
            slice_logger.info(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        else:
            # Load corrected data if not preprocessing
            with h5py.File(os.path.join(slice_output_dir, f"{slice_name}_corrected.h5"), 'r') as f:
                corrected_data = f['corrected_data'][:]
                image_shape = tuple(f['metadata/image_shape'][:])
            
            # Try to load CNMF components if they exist
            cnmf_components = None
            components_path = os.path.join(slice_output_dir, f"{slice_name}_cnmf_components.h5")
            if os.path.exists(components_path):
                slice_logger.info(f"Loading CNMF components from {components_path}")
                try:
                    with h5py.File(components_path, 'r') as f:
                        if 'cnmf_components' in f:
                            # Load components into dictionary
                            cnmf_components = {}
                            for key in f['cnmf_components'].keys():
                                cnmf_components[key] = f['cnmf_components'][key][:]
                            slice_logger.info(f"Loaded CNMF components: {list(cnmf_components.keys())}")
                except Exception as e:
                    slice_logger.warning(f"Failed to load CNMF components: {str(e)}")
        
        # ROI extraction
        if mode in ["all", "extract", "analyze"]:
            extraction_start = time.time()
            slice_logger.info("Starting ROI extraction")
            
            # Check if we should use the CNMF-aware ROI extraction function
            if cnmf_components is not None and "A" in cnmf_components and "C" in cnmf_components:
                # Import the CNMF-aware ROI extraction function
                from modules.roi_processing import extract_roi_fluorescence_with_cnmf
                
                slice_logger.info("Using CNMF-aware ROI extraction")
                # Get original image data for comparison if needed
                import tifffile
                original_data = None
                try:
                    with tifffile.TiffFile(tif_path) as tif:
                        original_data = tif.asarray()
                except Exception as e:
                    slice_logger.warning(f"Could not load original data for comparison: {str(e)}")
                
                # Use the CNMF-aware extraction function
                roi_masks, original_traces, refined_traces = extract_roi_fluorescence_with_cnmf(
                    roi_path,
                    original_data,
                    corrected_data,
                    image_shape,
                    slice_output_dir,
                    config["roi_processing"],
                    cnmf_components,
                    slice_logger
                )
                
                # Use the refined traces for further analysis
                roi_data = refined_traces
                
                # Also save the original traces for comparison
                if original_traces is not None:
                    comparison_file = os.path.join(slice_output_dir, f"{slice_name}_trace_comparison.h5")
                    with h5py.File(comparison_file, 'w') as f:
                        f.create_dataset('original_traces', data=original_traces)
                        f.create_dataset('refined_traces', data=refined_traces)
                        f.attrs['source_file'] = Path(tif_path).name
                    slice_logger.info(f"Saved trace comparison to {comparison_file}")
            else:
                # Use standard ROI extraction
                roi_masks, roi_data = extract_roi_fluorescence(
                    roi_path, 
                    corrected_data if 'corrected_data' in locals() else None,
                    image_shape,
                    slice_output_dir,
                    config["roi_processing"],
                    slice_logger
                )
            
            # Background subtraction - check if this is needed with CNMF data
            if cnmf_components is not None and config["roi_processing"]["background"].get("method", "") == "cnmf_background":
                slice_logger.info("Using CNMF background separation (skipping additional background subtraction)")
                bg_corrected_data = roi_data  # CNMF traces already have background removed
            else:
                slice_logger.info("Starting background subtraction")
                bg_corrected_data = subtract_background(
                    corrected_data if 'corrected_data' in locals() else None,
                    roi_data,
                    roi_masks,
                    config["roi_processing"]["background"],
                    slice_logger
                )
            
            # Save traces
            trace_file = os.path.join(slice_output_dir, f"{slice_name}_traces.h5")
            save_fluorescence_traces(trace_file, bg_corrected_data, metadata, roi_masks, config)
            
            extraction_time = time.time() - extraction_start
            slice_logger.info(f"ROI extraction completed in {extraction_time:.2f} seconds")
        else:
            # Load trace data if not extracting
            with h5py.File(os.path.join(slice_output_dir, f"{slice_name}_traces.h5"), 'r') as f:
                bg_corrected_data = f['bg_corrected_traces'][:]
                roi_masks = f['roi_masks'][:]
        
        # Fluorescence analysis
        if mode in ["all", "analyze"]:
            analysis_start = time.time()
            slice_logger.info("Starting fluorescence analysis")
            
            # Analyze fluorescence data
            metrics_df = analyze_fluorescence(
                bg_corrected_data, 
                roi_masks,
                tif_path,  # For original image
                config["analysis"],
                slice_logger
            )
            
            # Add metadata columns
            for key, value in metadata.items():
                metrics_df[key] = value
            
            # QC checks
            flagged_rois = perform_qc_checks(
                bg_corrected_data, 
                metrics_df, 
                config["analysis"]["qc_thresholds"],
                slice_logger
            )
            
            # Save metrics to Excel
            metrics_file = os.path.join(slice_output_dir, f"{slice_name}_metrics.xlsx")
            metrics_df.to_excel(metrics_file, index=False)
            
            # Generate verification visualizations
            slice_logger.info("Generating visualizations")
            generate_visualizations(
                bg_corrected_data,
                roi_masks,
                metrics_df,
                flagged_rois,
                tif_path,
                slice_output_dir,
                config["visualization"],
                slice_logger
            )
            
            analysis_time = time.time() - analysis_start
            slice_logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
            
            # Run advanced analysis if available and enabled
            advanced_results = None
            if ADVANCED_ANALYSIS_AVAILABLE and config.get("advanced_analysis", {}).get("enabled", False):
                slice_logger.info("Starting advanced analysis")
                advanced_start = time.time()
                
                advanced_results = run_advanced_analysis(
                    bg_corrected_data,
                    roi_masks,
                    metrics_df,
                    slice_output_dir,
                    config,
                    slice_logger
                )
                
                advanced_time = time.time() - advanced_start
                slice_logger.info(f"Advanced analysis completed in {advanced_time:.2f} seconds")
            
            return {
                "slice_name": slice_name,
                "metrics_file": metrics_file,
                "metadata": metadata,
                "flagged_rois": flagged_rois,
                "advanced_results": advanced_results,
                "timing": {
                    "preprocessing": preprocessing_time if 'preprocessing_time' in locals() else None,
                    "extraction": extraction_time if 'extraction_time' in locals() else None,
                    "analysis": analysis_time if 'analysis_time' in locals() else None,
                    "advanced": advanced_time if 'advanced_time' in locals() else None,
                    "total": time.time() - start_time
                }
            }
        
    except Exception as e:
        slice_logger.error(f"Error processing {slice_name}: {str(e)}", exc_info=True)
        return {
            "slice_name": slice_name,
            "error": str(e),
            "timing": {
                "total": time.time() - start_time
            }
        }

def extract_metadata_from_filename(filename):
    """Extract metadata from custom filename pattern 'CFA1_7.23.20_ipsi1_0um'."""
    # Initialize metadata dictionary
    metadata = {
        "mouse_id": "unknown",
        "date": "unknown",
        "pain_model": "unknown",
        "slice_type": "unknown",
        "slice_number": "1",
        "condition": "unknown"
    }
    
    # Split filename by underscore
    parts = filename.split('_')
    
    if len(parts) < 3:
        return metadata
    
    # First part typically contains pain model + mouse number (e.g., "CFA1")
    if parts[0]:
        # Extract pain model (letters) and mouse number (digits)
        model_match = re.match(r'([A-Za-z]+)(\d*)', parts[0])
        if model_match:
            metadata["pain_model"] = model_match.group(1)
            mouse_number = model_match.group(2) or "1"
            metadata["mouse_id"] = f"{metadata['pain_model']}{mouse_number}"
        else:
            metadata["mouse_id"] = parts[0]
    
    # Second part is usually the date
    if len(parts) > 1:
        metadata["date"] = parts[1]
    
    # Third part usually contains slice type and number
    if len(parts) > 2:
        # Look for ipsi/contra with optional number
        slice_match = re.match(r'(ipsi|contra)(\d*)', parts[2].lower())
        if slice_match:
            metadata["slice_type"] = slice_match.group(1).capitalize()  # Capitalize first letter
            metadata["slice_number"] = slice_match.group(2) or "1"
    
    # Last part usually has the condition
    for part in parts:
        if any(cond in part.lower() for cond in ["0um", "10um", "25um"]):
            metadata["condition"] = part
            break
    
    return metadata

def save_fluorescence_traces(filename, traces, metadata, roi_masks, config, cnmf_info=None):
    """Save fluorescence traces and metadata to HDF5 file."""
    with h5py.File(filename, 'w') as f:
        # Save traces
        f.create_dataset('bg_corrected_traces', data=traces)
        f.create_dataset('roi_masks', data=np.stack(roi_masks) if isinstance(roi_masks, list) else roi_masks)
        
        # Save CNMF info if available
        if cnmf_info is not None:
            cnmf_group = f.create_group('cnmf_info')
            cnmf_group.attrs['used_cnmf'] = True
            cnmf_group.attrs['method'] = cnmf_info.get('method', 'unknown')
            if 'components_file' in cnmf_info:
                cnmf_group.attrs['components_file'] = cnmf_info['components_file']
        
        # Save metadata
        meta_group = f.create_group('metadata')
        for key, value in metadata.items():
            if value is not None:
                meta_group.attrs[key] = value
        
        # Save configuration
        config_group = f.create_group('config')
        for section, params in config.items():
            section_group = config_group.create_group(section)
            for key, value in params.items():
                if isinstance(value, dict):
                    subsection = section_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if not isinstance(subvalue, (dict, list)):
                            try:
                                subsection.attrs[subkey] = subvalue
                            except:
                                # Some values might not be directly saveable to HDF5
                                subsection.attrs[subkey] = str(subvalue)
                elif not isinstance(value, (dict, list)):
                    try:
                        section_group.attrs[key] = value
                    except:
                        section_group.attrs[key] = str(value)
        
        # Save processing timestamp
        meta_group.attrs['processed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    """Main pipeline function."""
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config, args)
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Setup main logger
    main_log_file = os.path.join(output_dir, 'logs', 'main_pipeline.log')
    logger = setup_logging(main_log_file)
    
    logger.info("=" * 80)
    logger.info(f"Starting fluorescence pipeline in mode: {args.mode}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration file: {args.config}")
    logger.info("=" * 80)
    
    # Match .tif and .zip files
    start_time = time.time()
    logger.info("Matching .tif and .zip files")
    file_pairs = match_tif_and_roi_files(input_dir, logger)
    logger.info(f"Found {len(file_pairs)} matched file pairs")
    
    # Process file pairs in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_file_pair, 
                tif_path, 
                roi_path, 
                output_dir, 
                config, 
                args.mode,
                i
            ): (tif_path, roi_path) 
            for i, (tif_path, roi_path) in enumerate(file_pairs)
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slices"):
            tif_path, roi_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed processing {Path(tif_path).stem}")
            except Exception as e:
                logger.error(f"Error processing {Path(tif_path).stem}: {str(e)}")
    
    # Generate mouse-level summaries
    if args.mode in ["all", "analyze"]:
        logger.info("Generating mouse-level summaries")
        
        # Group results by mouse ID
        mouse_data = {}
        for result in results:
            if "error" in result:
                continue
                
            mouse_id = result["metadata"]["mouse_id"]
            if mouse_id not in mouse_data:
                mouse_data[mouse_id] = []
            mouse_data[mouse_id].append(result)
        
        # Create summary file for each mouse
        for mouse_id, slices in mouse_data.items():
            save_mouse_summary(mouse_id, slices, output_dir, logger)
    
    # Save pipeline summary
    total_time = time.time() - start_time
    summary = {
        "pipeline_mode": args.mode,
        "total_time": total_time,
        "processed_slices": len(results),
        "successful_slices": sum(1 for r in results if "error" not in r),
        "failed_slices": sum(1 for r in results if "error" in r),
        "advanced_analysis_available": ADVANCED_ANALYSIS_AVAILABLE,
        "advanced_analysis_enabled": config.get("advanced_analysis", {}).get("enabled", False),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Pipeline completed in {total_time:.2f} seconds")
    logger.info(f"Processed {len(results)} slices:")
    logger.info(f"  - Successful: {summary['successful_slices']}")
    logger.info(f"  - Failed: {summary['failed_slices']}")
    if ADVANCED_ANALYSIS_AVAILABLE and config.get("advanced_analysis", {}).get("enabled", False):
        logger.info("Advanced analysis was performed")
    logger.info(f"Summary saved to {os.path.join(output_dir, 'pipeline_summary.json')}")

if __name__ == "__main__":
    main()