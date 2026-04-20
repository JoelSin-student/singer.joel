"""
Synchronize extracted Awinda tabs to match the insole/skeleton time grid.

This module ensures that Awinda target files (positions and angles) have the same
row count and temporal alignment as the corresponding insole/skeleton pairs.

Key insight: Extracted Awinda tabs from xlsx files come with Frame numbers but no
timestamp information. They need to be resampled onto the same 60Hz grid that the
insole files were synchronized to.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def synchronize_awinda_tabs_to_insole_grid(
    clean_data_dir,
    insole_dir,
    awinda_required_tab_dir,
    output_dir,
    sampling_hz=60.0,
):
    """
    Synchronize extracted Awinda tabs by truncating/padding to match insole row counts.
    
    This function ensures that extracted Awinda tabs have the same row count as the 
    synchronized insole files they correspond to.
    
    Args:
        clean_data_dir: Path to clean_data directory (contains Soles_*.txt files)
        insole_dir: Path to directory with Soles_*.txt files (generally same as clean_data_dir)
        awinda_required_tab_dir: Path to extracted tabs (position/angles)
        output_dir: Where to save synchronized tab CSVs
        sampling_hz: Sampling frequency (default 60.0 Hz, not directly used for sync)
    """
    clean_data_dir = Path(clean_data_dir)
    insole_dir = Path(insole_dir)
    awinda_required_tab_dir = Path(awinda_required_tab_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all insole files (cleaned, synchronized versions)
    insole_files = sorted(clean_data_dir.glob('Soles_*.txt'))
    if not insole_files:
        raise FileNotFoundError(f"No insole files found in {clean_data_dir}")
    
    def extract_tag(filename, prefix):
        """Extract tag from filename"""
        name = filename.stem
        if name.startswith(prefix + '_'):
            return name.split('_', 1)[1]
        return None
    
    def read_awinda_tab(path):
        """Read Awinda tab with flexible delimiter"""
        df = pd.read_csv(path, sep=";", engine="python")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df
    
    def find_single_tab_csv(tag, pattern):
        """Find extracted tab file matching tag and pattern"""
        matches = sorted(awinda_required_tab_dir.glob(f"Awinda_{tag}_{pattern}.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one match for tag={tag}, pattern={pattern}, "
                f"found {len(matches)}"
            )
        return matches[0]
    
    # Process each insole file
    synchronized_count = 0
    for insole_path in insole_files:
        tag = extract_tag(insole_path, 'Soles')
        if not tag:
            continue
        
        try:
            # Read insole file to get the target row count
            insole_df = pd.read_csv(insole_path, sep='\t')
            target_n_rows = len(insole_df)
            
            # Find and read extracted tabs
            pos_path = find_single_tab_csv(tag, "*Segment_Position*")
            ang_path = find_single_tab_csv(tag, "*Joint_Angles_ZXY*")
            
            pos_tab = read_awinda_tab(pos_path)
            ang_tab = read_awinda_tab(ang_path)
            
            # Synchronize: truncate/pad to match insole row count
            # (keep first n_rows from each tab)
            pos_sync = pos_tab.iloc[:target_n_rows].copy()
            ang_sync = ang_tab.iloc[:target_n_rows].copy()
            
            # If tabs are shorter than insole, pad with forward-fill
            if len(pos_sync) < target_n_rows:
                pos_sync = pos_sync.reindex(range(target_n_rows)).ffill()
            if len(ang_sync) < target_n_rows:
                ang_sync = ang_sync.reindex(range(target_n_rows)).ffill()
            
            # Save synchronized tabs
            pos_out = output_dir / pos_path.name
            ang_out = output_dir / ang_path.name
            pos_sync.to_csv(pos_out, index=False)
            ang_sync.to_csv(ang_out, index=False)
            
            print(
                f"[sync] {tag}: "
                f"position {len(pos_tab)} -> {len(pos_sync)}, "
                f"angles {len(ang_tab)} -> {len(ang_sync)}"
            )
            synchronized_count += 1
            
        except Exception as e:
            print(f"[error] {tag}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSynchronized {synchronized_count} Awinda tab pairs to insole row count.")
    return output_dir
