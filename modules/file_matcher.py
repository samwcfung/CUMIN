#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Matcher Module (Modified for Custom Filenames)
------------------
Matches .tif video files with their corresponding .zip ROI files.
Adapted for filenames like 'CFA1_7.23.20_ipsi1_0um'
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def match_tif_and_roi_files(input_dir: str, logger) -> List[Tuple[str, str]]:
    """
    Match .tif video files with their corresponding .zip ROI files.
    
    Parameters
    ----------
    input_dir : str
        Directory containing .tif and .zip files
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    List[Tuple[str, str]]
        List of tuples containing matched (tif_path, roi_path)
    """
    input_path = Path(input_dir)
    
    # Find all .tif and .zip files
    tif_files = list(input_path.glob("**/*.tif"))
    zip_files = list(input_path.glob("**/*.zip"))
    
    logger.info(f"Found {len(tif_files)} .tif files and {len(zip_files)} .zip files")
    
    # Group files by experiment condition (_0um, _10um, _25um)
    conditions = ["_0um", "_10um", "_25um"]
    
    # Dictionary to store matched files
    matched_pairs = []
    unmatched_tifs = []
    
    # Match files based on naming patterns
    for tif_file in tif_files:
        tif_name = tif_file.stem
        found_match = False
        
        # Try to find exact match first (same name but .zip extension)
        exact_match = next((zip_file for zip_file in zip_files 
                          if zip_file.stem == tif_name), None)
        
        if exact_match:
            matched_pairs.append((str(tif_file), str(exact_match)))
            found_match = True
            logger.debug(f"Exact match found for {tif_name}")
            continue
        
        # Check for condition-based matches
        for condition in conditions:
            if condition in tif_name.lower():
                # Extract base name without condition
                base_name = tif_name.replace(condition, "")
                
                # Match ipsi/contra with numeric identifiers
                # Extract the slice type pattern (e.g., "ipsi1", "contra2")
                slice_type_pattern = None
                for slice_type in ["ipsi", "contra"]:
                    if slice_type in tif_name.lower():
                        # Look for numeric suffix after the slice type
                        match = re.search(f"{slice_type}\\d*", tif_name.lower(), re.IGNORECASE)
                        if match:
                            slice_type_pattern = match.group(0)
                            break
                        else:
                            slice_type_pattern = slice_type
                            break
                
                # Find matching ROI file with same base pattern
                for zip_file in zip_files:
                    zip_name = zip_file.stem
                    
                    # Check if this zip file has the same condition
                    if condition not in zip_name.lower():
                        continue
                        
                    # Check if slice type matches
                    if slice_type_pattern and slice_type_pattern not in zip_name.lower():
                        continue
                    
                    # Check if base parts match - we'll consider it a match if enough elements are common
                    # Split filenames by underscore to compare components
                    tif_parts = tif_name.lower().split('_')
                    zip_parts = zip_name.lower().split('_')
                    
                    # Count matching elements
                    common_elements = set(tif_parts).intersection(set(zip_parts))
                    
                    # If enough elements match, consider it a pair
                    if len(common_elements) >= len(tif_parts) - 1:  # Allow one element to differ
                        matched_pairs.append((str(tif_file), str(zip_file)))
                        found_match = True
                        logger.debug(f"Matched {tif_name} -> {zip_name}")
                        break
                
                if found_match:
                    break
        
        if not found_match:
            unmatched_tifs.append(tif_name)
    
    # Log matching results
    logger.info(f"Successfully matched {len(matched_pairs)} file pairs")
    if unmatched_tifs:
        logger.warning(f"Could not find matching ROI files for {len(unmatched_tifs)} .tif files")
        for unmatched in unmatched_tifs[:5]:  # Log first 5 unmatched files
            logger.warning(f"  - {unmatched}")
        if len(unmatched_tifs) > 5:
            logger.warning(f"  - ... and {len(unmatched_tifs) - 5} more")
    
    return matched_pairs

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from custom filename format 'CFA1_7.23.20_ipsi1_0um'.
    
    Parameters
    ----------
    filename : str
        The filename to analyze
        
    Returns
    -------
    Dict[str, str]
        Dictionary of extracted metadata
    """
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

def group_files_by_mouse(file_pairs: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group matched file pairs by mouse ID.
    
    Parameters
    ----------
    file_pairs : List[Tuple[str, str]]
        List of matched (tif_path, roi_path) pairs
    
    Returns
    -------
    Dict[str, List[Tuple[str, str]]]
        Dictionary mapping mouse IDs to lists of file pairs
    """
    mouse_groups = {}
    
    for tif_path, roi_path in file_pairs:
        # Extract metadata from filename
        tif_name = Path(tif_path).stem
        metadata = extract_metadata_from_filename(tif_name)
        
        mouse_id = metadata.get("mouse_id", "unknown")
        
        if mouse_id not in mouse_groups:
            mouse_groups[mouse_id] = []
        
        mouse_groups[mouse_id].append((tif_path, roi_path))
    
    return mouse_groups