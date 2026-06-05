"""
Align extracted Awinda tabs to the insole row grid.

This module ensures that Awinda target files (positions and angles) have the same
row count as the corresponding preprocessed insole/skeleton pairs.

Important: this is not physical resampling of Awinda signals. We only align row
counts with strict tail-trim semantics, because insole files are the time
reference after preprocessing.
"""

import pandas as pd
from pathlib import Path


def _count_data_rows(path):
    """Count data rows quickly without parsing the full table."""
    path = Path(path)
    with path.open("rb") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def _align_rows_to_target_count(df, target_n_rows):
    """Align row count with strict tail-trim semantics.

    Rules:
    - If source == target: return unchanged copy.
    - If source > target: keep the first target rows and trim only the tail.
    - If source < target: raise (cannot tail-trim to add rows).
    """
    source_n_rows = int(len(df))
    target_n_rows = int(target_n_rows)

    if source_n_rows <= 0 or target_n_rows <= 0:
        raise ValueError(f"Invalid row counts source={source_n_rows}, target={target_n_rows}")
    if source_n_rows == target_n_rows:
        return df.copy()

    if source_n_rows < target_n_rows:
        raise ValueError(
            f"Awinda rows ({source_n_rows}) are shorter than target insole rows ({target_n_rows}); "
            "strict tail-trim cannot increase row count."
        )

    return df.iloc[:target_n_rows].reset_index(drop=True).copy()


def synchronize_awinda_tabs_to_insole_grid(
    clean_data_dir,
    insole_dir,
    awinda_required_tab_dir,
    output_dir,
    sampling_hz=60.0,
):
    """
    Align extracted Awinda tabs to match insole row counts.
    
    This function ensures that extracted Awinda tabs have the same row count as the
    synchronized insole files they correspond to.
    
    Args:
        clean_data_dir: Path to clean_data directory (contains Soles_*.txt files)
        insole_dir: Path to directory with Soles_*.txt files (generally same as clean_data_dir)
        awinda_required_tab_dir: Path to extracted tabs (position/angles)
        output_dir: Where to save synchronized tab CSVs
        sampling_hz: Sampling frequency (unused; kept for backward compatibility)
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
    skipped_count = 0
    for insole_path in insole_files:
        tag = extract_tag(insole_path, 'Soles')
        if not tag:
            continue
        
        try:
            # Find and read extracted tabs
            pos_path = find_single_tab_csv(tag, "*Segment_Position*")
            ang_path = find_single_tab_csv(tag, "*Joint_Angles_ZXY*")

            # Fast row-count path: full parse is unnecessary for synchronization.
            target_n_rows = _count_data_rows(insole_path)

            pos_out = output_dir / pos_path.name
            ang_out = output_dir / ang_path.name

            newest_input_mtime = max(
                insole_path.stat().st_mtime,
                pos_path.stat().st_mtime,
                ang_path.stat().st_mtime,
            )
            outputs_ready = (
                pos_out.is_file()
                and ang_out.is_file()
                and min(pos_out.stat().st_mtime, ang_out.stat().st_mtime) >= newest_input_mtime
            )

            if outputs_ready:
                pos_rows = _count_data_rows(pos_out)
                ang_rows = _count_data_rows(ang_out)
                if pos_rows == target_n_rows and ang_rows == target_n_rows:
                    print(f"[sync:skip] {tag}: outputs already up to date ({target_n_rows} rows)")
                    skipped_count += 1
                    continue
            
            pos_tab = read_awinda_tab(pos_path)
            ang_tab = read_awinda_tab(ang_path)
            
            # Row-count alignment only (no value interpolation).
            pos_sync = _align_rows_to_target_count(pos_tab, target_n_rows)
            ang_sync = _align_rows_to_target_count(ang_tab, target_n_rows)
            
            # Save synchronized tabs
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
    if skipped_count:
        print(f"Skipped {skipped_count} already synchronized pairs.")
    return output_dir
