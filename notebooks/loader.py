# Loader-related functions
#
#
#
#
import pandas as pd
import numpy as np
import os
import glob
import yaml
import time
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

def load_config(args,config_path, model):
    """Load a YAML file from the given path and return it as a Python object.
    Args:
        path (str): File path of the YAML configuration file to load.
    Returns:
        dict: Dictionary converted from YAML content.
    """
    # If config_path is not provided, infer it from model name and mode.
    if config_path is None:
        config_path = f'notebooks/config/{model}/{args.mode}.yaml'
    
    # Print the config file currently in use.
    print(f"<load config>")
    print(f"Loading configuration from: {config_path}")
    time.sleep(0.5)

    # Load config file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config values with command-line arguments.
    # Priority: mode section (train/predict/visual) -> location section -> top-level.
    cli_args = vars(args)   # Convert args Namespace into a dictionary
    override_log = []
    mode_section = args.mode if isinstance(config.get(args.mode), dict) else None
    skip_keys = {"mode", "model", "config"}

    for key, value in cli_args.items():
        if key in skip_keys or value is None:
            continue

        if mode_section and key in config[mode_section]:
            previous = config[mode_section][key]
            config[mode_section][key] = value
            override_log.append(f"{mode_section}.{key}: {previous} -> {value}")
            continue

        if isinstance(config.get("location"), dict) and key in config["location"]:
            previous = config["location"][key]
            config["location"][key] = value
            override_log.append(f"location.{key}: {previous} -> {value}")
            continue

        if key in config:
            previous = config[key]
            config[key] = value
            override_log.append(f"{key}: {previous} -> {value}")
            continue

        # Keep unknown CLI keys accessible for downstream custom code.
        config[key] = value
        override_log.append(f"{key}: <new> -> {value}")

    if override_log:
        print("Applied CLI overrides:")
        for line in override_log:
            print(f"- {line}")

    return config


def get_datapath_pairs(skeleton_dir, insole_dir):
    """Pair skeleton/insole file paths sharing the same tag from directories.
    Args:
        skeleton_dir (str): Directory containing skeleton files (Awinda_*.csv).
        insole_dir (str): Directory containing insole files (Soles_*.txt).
    Returns:
        defaultdict: Dictionary keyed by tag.
            Value format: `{tag:{'skeleton': str, 'insole': str}}`
    """
    # Show data paths
    print("---"*20)
    print(f"<Dataset Infomation>")
    print(f"skeleton data path : {skeleton_dir}")
    print(f"Inosole data path : {insole_dir}")
    time.sleep(0.5)

    # Get all CSV files in each folder
    skeleton_files = glob.glob(os.path.join(skeleton_dir, "Awinda_*.csv"))
    insole_files = glob.glob(os.path.join(insole_dir, "Soles_*.txt"))

    # Create dictionary to store skeleton/insole data pairs
    data_pairs = defaultdict(lambda: {'skeleton': None, 'insole': None})

    # Extract tags from skeleton files and store them
    for file_path in skeleton_files:
        filename = os.path.basename(file_path)
        tag = filename.split('_', 1)[1].rsplit('.', 1)[0]  # Extract tag between first underscore and file extension
        data_pairs[tag]['skeleton'] = file_path
    
    # Extract tags from insole files and store them
    for file_path in insole_files:
        filename = os.path.basename(file_path)
        tag = filename.split('_', 1)[1].rsplit('.', 1)[0]

        # Add only when the matching skeleton file exists
        if tag in data_pairs:
            if data_pairs[tag]['insole'] is None:
                data_pairs[tag]['insole'] = file_path
            else:
                raise ValueError(
                    f"Duplicate insole file found for tag '{tag}': "
                    f"'{data_pairs[tag]['insole']}' and '{file_path}'"
                )

    # Print extracted pairing results
    data_i=0
    for tag, paths in data_pairs.items():
        data_i+=1
        print(f"")
        print(f"Data_{data_i}_{tag}")
        print(f"skeleton: {paths['skeleton']}")
        print(f"insole: {paths['insole']}")
        time.sleep(0.3)
    print("---"*20)

    return data_pairs


def load_and_combine_data(data_pairs):
    """Load data from file-path dict and return concatenated DataFrames.
    Args:
        data_pairs (dict): Object with tags as keys and file-path dictionaries as values.
    Returns:
        tuple (pd.DataFrame, pd.DataFrame, np.ndarray):
            Concatenated skeleton/insole DataFrames and a per-frame segment id array.
    """
    # Create lists to store each data category
    all_skeleton_df = []
    all_insole_df   = []
    all_segment_ids = []
    
    # Load each file as DataFrame and append to lists
    for segment_id, (tag, paths) in enumerate(sorted(data_pairs.items()), start=1):
        if paths['skeleton'] is None:
            raise ValueError(f"Missing skeleton file for tag '{tag}'")

        if paths['insole'] is None:
            raise ValueError(f"Missing insole file for tag '{tag}'")

        skeleton_df = pd.read_csv(paths['skeleton'])
        # Drop non-coordinate/time index columns from skeleton targets.
        skeleton_df = skeleton_df.drop(
            columns=['Frame', 'frame', '# time', 'time', 'Timestamp'],
            errors='ignore'
        )
        
        # Insole files may be comma- or tab-delimited; do not strip header with comment markers.
        insole_df = pd.read_csv(paths['insole'], engine='python')
        insole_df.columns = insole_df.columns.str.strip()

        if len(skeleton_df) != len(insole_df):
            raise ValueError(
                f"Length mismatch for tag '{tag}': "
                f"skeleton has {len(skeleton_df)} rows but insole has {len(insole_df)} rows. "
                "Please rerun preprocessing synchronization before training."
            )

        all_skeleton_df.append(skeleton_df)
        all_insole_df.append(insole_df)
        all_segment_ids.append(np.full(len(skeleton_df), segment_id, dtype=np.int32))

    if not all_skeleton_df or not all_insole_df:
        raise ValueError("No valid skeleton/insole pairs were found to load.")

    return (
        pd.concat(all_skeleton_df, ignore_index=True),
        pd.concat(all_insole_df, ignore_index=True),
        np.concatenate(all_segment_ids, axis=0),
    )


def _read_awinda_tab_csv(file_path):
    """Read converted Awinda tab CSV files with delimiter fallback and numeric coercion."""
    df = pd.read_csv(file_path, sep=';', engine='python')
    if df.shape[1] == 1:
        # Some exports are comma-delimited (e.g., extracted from pandas to_csv default).
        df = pd.read_csv(file_path, sep=',', engine='python')

    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Frame", "frame", "# time", "time", "Timestamp"], errors='ignore')

    # Keep only numeric channels for model targets.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df.shape[1] == 0:
        raise ValueError(f"No usable numeric columns found in Awinda tab file: {file_path}")

    return df


def _resolve_awinda_tab_path(awinda_tabs_dir, tag, suffix, fallback_pattern=None):
    preferred = os.path.join(awinda_tabs_dir, f"Awinda_{tag}_{suffix}.csv")
    if os.path.isfile(preferred):
        return preferred

    candidates = glob.glob(os.path.join(awinda_tabs_dir, f"*_{tag}_{suffix}.csv"))
    if len(candidates) == 1:
        return candidates[0]

    if fallback_pattern:
        candidates = glob.glob(os.path.join(awinda_tabs_dir, f"Awinda_{tag}_{fallback_pattern}.csv"))
        if len(candidates) != 1:
            candidates = glob.glob(os.path.join(awinda_tabs_dir, f"*_{tag}_{fallback_pattern}.csv"))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple Awinda tab files found for tag '{tag}' and suffix/pattern '{suffix}': {candidates}"
        )

    raise FileNotFoundError(
        f"Awinda converted tab file not found for tag '{tag}' with suffix/pattern '{suffix}' in {awinda_tabs_dir}"
    )


def load_awinda_targets_from_converted_tabs(
    data_pairs,
    awinda_tabs_dir,
    include_positions=True,
    include_joint_angles=True,
    joint_angles_suffix='tab9_Joint_Angles_ZXY',
):
    """Load and concatenate Awinda targets from converted tab CSV files.

    Args:
        data_pairs (dict): skeleton/insole pairing dictionary keyed by tag.
        awinda_tabs_dir (str): Directory containing converted Awinda tab CSV files.
        include_positions (bool): Include tab4 segment positions.
        include_joint_angles (bool): Include joint-angle tab columns.
        joint_angles_suffix (str): Tab suffix for angles, e.g. tab9_Joint_Angles_ZXY.

    Returns:
        tuple(pd.DataFrame, dict): Concatenated targets and metadata.
    """
    if not include_positions and not include_joint_angles:
        raise ValueError("At least one target source must be enabled for Awinda target loading.")

    all_targets = []
    meta = {
        "position_columns": [],
        "angle_columns": [],
        "joint_angles_suffix": joint_angles_suffix,
    }
    expected_columns = None

    for _, (tag, _) in enumerate(sorted(data_pairs.items()), start=1):
        target_parts = []

        if include_positions:
            pos_path = _resolve_awinda_tab_path(
                awinda_tabs_dir,
                tag,
                'tab4_Segment_Position',
                fallback_pattern='*Segment_Position*',
            )
            pos_df = _read_awinda_tab_csv(pos_path)
            pos_df = pos_df.add_prefix('pos::')
            target_parts.append(pos_df)
            if not meta["position_columns"]:
                meta["position_columns"] = list(pos_df.columns)

        if include_joint_angles:
            angle_pattern = f"*{joint_angles_suffix}*" if "*" not in joint_angles_suffix else joint_angles_suffix
            angle_candidates = glob.glob(os.path.join(awinda_tabs_dir, f"Awinda_{tag}_{angle_pattern}.csv"))
            if len(angle_candidates) != 1:
                angle_candidates = glob.glob(os.path.join(awinda_tabs_dir, f"*_{tag}_{angle_pattern}.csv"))

            # Exclude ergonomic-angle tabs by default to keep a single canonical joint-angle target.
            angle_candidates = [
                p for p in angle_candidates
                if "ergonomic_joint_angles" not in os.path.basename(p).lower()
            ]

            if len(angle_candidates) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one non-ergonomic angle tab for tag '{tag}' with pattern "
                    f"'{angle_pattern}' in {awinda_tabs_dir}, found {len(angle_candidates)}"
                )

            ang_path = angle_candidates[0]
            ang_df = _read_awinda_tab_csv(ang_path)
            ang_df = ang_df.add_prefix('ang::')
            target_parts.append(ang_df)
            if not meta["angle_columns"]:
                meta["angle_columns"] = list(ang_df.columns)

        if not target_parts:
            continue

        combined_df = pd.concat(target_parts, axis=1)

        if expected_columns is None:
            expected_columns = list(combined_df.columns)
        elif list(combined_df.columns) != expected_columns:
            raise ValueError(
                f"Awinda target schema mismatch for tag '{tag}'. "
                f"Expected {len(expected_columns)} columns but got {combined_df.shape[1]}."
            )

        all_targets.append(combined_df)

    if not all_targets:
        raise ValueError("No Awinda targets were loaded from converted tab CSV files.")

    meta["target_columns"] = list(all_targets[0].columns)
    return pd.concat(all_targets, ignore_index=True), meta


def load_awinda_targets_from_merged_csv(data_pairs, awinda_targets_dir):
    """Load premerged Awinda target CSVs (AwindaTarget_<tag>.csv) for soleformer.

    Args:
        data_pairs (dict): skeleton/insole pairing dictionary keyed by tag.
        awinda_targets_dir (str): Directory containing merged target files.

    Returns:
        tuple(pd.DataFrame, dict): Concatenated targets and metadata.
    """
    all_targets = []
    expected_columns = None
    awinda_targets_dir = str(awinda_targets_dir)

    for _, (tag, _) in enumerate(sorted(data_pairs.items()), start=1):
        target_path = os.path.join(awinda_targets_dir, f"AwindaTarget_{tag}.csv")
        if not os.path.isfile(target_path):
            raise FileNotFoundError(
                f"Merged Awinda target file not found for tag '{tag}': {target_path}"
            )

        df = pd.read_csv(target_path, sep=';', engine='python')
        if df.shape[1] == 1:
            df = pd.read_csv(target_path, sep=',', engine='python')

        df.columns = df.columns.str.strip()
        df = df.drop(columns=['Frame', 'frame', '# time', 'time', 'Timestamp'], errors='ignore')

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.shape[1] == 0:
            raise ValueError(f"No usable numeric target columns found in {target_path}")

        if expected_columns is None:
            expected_columns = list(df.columns)
        elif list(df.columns) != expected_columns:
            raise ValueError(
                f"Awinda merged target schema mismatch for tag '{tag}'. "
                f"Expected {len(expected_columns)} columns but got {df.shape[1]}."
            )

        all_targets.append(df)

    if not all_targets:
        raise ValueError("No Awinda merged targets were loaded.")

    meta = {
        "target_columns": list(all_targets[0].columns),
        "position_columns": [c for c in all_targets[0].columns if c.startswith('pos::')],
        "angle_columns": [c for c in all_targets[0].columns if c.startswith('ang::')],
        "joint_angles_suffix": "merged_csv",
    }

    return pd.concat(all_targets, ignore_index=True), meta


def restructure_insole_data(insole_df):
    """Split insole data into pressure/IMU/time feature.
    Args:
        insole_df (pd.DataFrame): DataFrame containing insole data.
    Returns:
        tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            Pressure features, IMU features, and one-column time feature.
    """
    insole_df = insole_df.copy()
    insole_df.columns = insole_df.columns.str.strip()

    # Normalize time-column naming across pipelines.
    if '# time' not in insole_df.columns and 'time' in insole_df.columns:
        insole_df = insole_df.rename(columns={'time': '# time'})

    imu_cols = [
        "left acceleration X[g]", "left acceleration Y[g]", "left acceleration Z[g]",
        "left angular X[dps]", "left angular Y[dps]", "left angular Z[dps]",
        "right acceleration X[g]", "right acceleration Y[g]", "right acceleration Z[g]",
        "right angular X[dps]", "right angular Y[dps]", "right angular Z[dps]"
    ]

    missing_imu = [c for c in imu_cols if c not in insole_df.columns]
    if missing_imu:
        raise KeyError(
            f"Required IMU columns were not found. Missing: {missing_imu}. "
            f"Available columns start with: {list(insole_df.columns[:10])}"
        )

    # Build an explicit time feature so models can learn non-periodic drift inside windows.
    if "# time" in insole_df.columns:
        time_series = pd.to_numeric(insole_df["# time"], errors="coerce")
    elif "time" in insole_df.columns:
        time_series = pd.to_numeric(insole_df["time"], errors="coerce")
    elif "Timestamp" in insole_df.columns:
        time_series = pd.to_numeric(insole_df["Timestamp"], errors="coerce")
    elif "Frame" in insole_df.columns:
        time_series = pd.to_numeric(insole_df["Frame"], errors="coerce")
    elif "frame" in insole_df.columns:
        time_series = pd.to_numeric(insole_df["frame"], errors="coerce")
    else:
        time_series = pd.Series(np.arange(len(insole_df), dtype=np.float32), index=insole_df.index)

    time_series = time_series.bfill().ffill()
    if time_series.isna().any():
        time_series = pd.Series(np.arange(len(insole_df), dtype=np.float32), index=insole_df.index)

    time_feature = pd.DataFrame({"time_feature": time_series.astype(np.float32)})

    drop_from_pressure = [
        "# time",
        "Frame",  # Remove frame index; not a real measurement
        *imu_cols,
        "left total force[N]", "left center of pressure X[-0.5...+0.5]", "left center of pressure Y[-0.5...+0.5]",
        "right total force[N]", "right center of pressure X[-0.5...+0.5]", "right center of pressure Y[-0.5...+0.5]",
        "right steps[]", "left steps[]"
    ]

    pressure_lr = insole_df.drop(columns=drop_from_pressure, errors='ignore')
    IMU_lr = insole_df[imu_cols]

    return pressure_lr, IMU_lr, time_feature


def normalize_time_feature_per_segment(time_feature_df, segment_ids):
    """Normalize time feature independently within each segment to [0, 1].
    
    This ensures that temporal progression is learned relative to each segment's
    local time range, avoiding encoding of global recording offsets.
    
    Args:
        time_feature_df (pd.DataFrame): DataFrame with one 'time_feature' column.
        segment_ids (np.ndarray): Array of segment IDs for each frame, matching length.
    
    Returns:
        np.ndarray: Normalized time feature (same shape, float32).
    """
    time_array = time_feature_df.values.astype(np.float32).ravel()
    normalized = np.zeros_like(time_array)
    
    for seg_id in np.unique(segment_ids):
        mask = segment_ids == seg_id
        seg_min = time_array[mask].min()
        seg_max = time_array[mask].max()
        seg_range = seg_max - seg_min
        
        if seg_range > 1e-8:
            # Normalize segment to [0, 1]
            normalized[mask] = (time_array[mask] - seg_min) / seg_range
        else:
            # Handle constant time within segment (e.g., single frame or no time variation)
            normalized[mask] = 0.0
    
    return normalized.reshape(-1, 1).astype(np.float32)


def calculate_grad(
    pressure_lr,
    IMU_lr,
    window_length=5,
    polyorder=2,
    smooth_grad1=False,
    normalization_stats=None,
    return_stats=False,
):
    """Compute smoothed 1st/2nd derivatives, concatenate, and normalize.
    Args:
        pressure_lr (array-like): Pressure-sensor time-series data.
        IMU_lr (array-like): IMU time-series data.
        window_length (int): Savitzky-Golay filter window length (must be odd).
        polyorder (int): Polynomial order for Savitzky-Golay filter.
        smooth_grad1 (bool): Whether to smooth first-derivative signals before second derivative.
        normalization_stats (dict | None): Optional pre-fitted mean/std stats for pressure and IMU features.
        return_stats (bool): When True, also return fitted normalization statistics.
    Returns:
        tuple: Normalized expanded feature arrays, optionally followed by fitted stats.
    """
    # Convert to numpy arrays
    pressure_arr = np.asarray(pressure_lr, dtype=np.float32)
    IMU_arr = np.asarray(IMU_lr, dtype=np.float32)

    # Make SG settings robust for short sequences and invalid combinations.
    window_length = int(window_length)
    polyorder = int(polyorder)
    n_frames = pressure_arr.shape[0]
    if n_frames < 3:
        raise ValueError("Not enough frames to compute gradient features (need at least 3).")
    if window_length < 3:
        window_length = 3
    if window_length % 2 == 0:
        window_length += 1
    if window_length > n_frames:
        window_length = n_frames if n_frames % 2 == 1 else n_frames - 1
    polyorder = max(1, min(polyorder, window_length - 1))

    # Smooth before each derivative stage using Savitzky-Golay filter
    pressure_smooth = savgol_filter(pressure_arr, window_length=window_length, polyorder=polyorder, axis=0, mode='nearest')
    pressure_grad1 = np.gradient(pressure_smooth, axis=0)
    if smooth_grad1:
        pressure_grad1_for_grad2 = savgol_filter(pressure_grad1, window_length=window_length, polyorder=polyorder, axis=0, mode='nearest')
    else:
        pressure_grad1_for_grad2 = pressure_grad1
    pressure_grad2 = np.gradient(pressure_grad1_for_grad2, axis=0)

    IMU_smooth = savgol_filter(IMU_arr, window_length=window_length, polyorder=polyorder, axis=0, mode='nearest')
    IMU_grad1 = np.gradient(IMU_smooth, axis=0)
    if smooth_grad1:
        IMU_grad1_for_grad2 = savgol_filter(IMU_grad1, window_length=window_length, polyorder=polyorder, axis=0, mode='nearest')
    else:
        IMU_grad1_for_grad2 = IMU_grad1
    IMU_grad2 = np.gradient(IMU_grad1_for_grad2, axis=0)

    # Concatenate original + derivatives
    pressure_features = np.concatenate([pressure_arr, pressure_grad1, pressure_grad2], axis=1)
    IMU_features = np.concatenate([IMU_arr, IMU_grad1, IMU_grad2], axis=1)

    if normalization_stats is None:
        pressure_mean = pressure_features.mean(axis=0)
        pressure_std = pressure_features.std(axis=0)
        imu_mean = IMU_features.mean(axis=0)
        imu_std = IMU_features.std(axis=0)
    else:
        pressure_mean = np.asarray(normalization_stats["pressure_mean"], dtype=np.float32)
        pressure_std = np.asarray(normalization_stats["pressure_std"], dtype=np.float32)
        imu_mean = np.asarray(normalization_stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(normalization_stats["imu_std"], dtype=np.float32)

    # Z-score normalization: (x - mean) / std per feature
    pressure_features = (pressure_features - pressure_mean) / (pressure_std + 1e-8)
    IMU_features = (IMU_features - imu_mean) / (imu_std + 1e-8)

    if return_stats:
        fitted_stats = {
            "pressure_mean": pressure_mean.astype(np.float32),
            "pressure_std": pressure_std.astype(np.float32),
            "imu_mean": imu_mean.astype(np.float32),
            "imu_std": imu_std.astype(np.float32),
        }
        return pressure_features, IMU_features, fitted_stats

    return pressure_features, IMU_features
    
    
# Debug-friendly PressureSkeletonDataset
class PressureSkeletonDataset(Dataset):
    """PyTorch custom dataset for pressure and skeleton data.

    Args:
        input_feature (array-like): Sequence of pressure and IMU features.
        skeleton_data (array-like): Sequence of corresponding skeleton data.
        sequence_length (int): Number of frames fed to the model.

    Returns:
        tuple (torch.Tensor, torch.Tensor):
            - Input tensor with shape sequence_len x feature_dim
            - Target tensor with shape feature_dim_skeleton
    """

    def __init__(self, input_feature, skeleton_data, sequence_length, segment_ids=None):
        self.sequence_length = int(sequence_length)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")

        self.input_data = np.asarray(input_feature, dtype=np.float32)
        self.skeleton_data = np.asarray(skeleton_data, dtype=np.float32)
        self.segment_ids = None if segment_ids is None else np.asarray(segment_ids)

        if len(self.input_data) != len(self.skeleton_data):
            raise ValueError(
                "input_feature and skeleton_data must contain the same number of frames"
            )
        if self.segment_ids is not None and len(self.segment_ids) != len(self.input_data):
            raise ValueError("segment_ids length must match input_feature length")

        self._valid_length = len(self.input_data) - self.sequence_length + 1
        if self._valid_length <= 0:
            raise ValueError(
                "Not enough frames to build a sequence. Increase data length or decrease sequence_length"
            )

        if self.segment_ids is None:
            self.valid_starts = np.arange(self._valid_length, dtype=np.int64)
        else:
            start_segment = self.segment_ids[:self._valid_length]
            end_segment = self.segment_ids[self.sequence_length - 1:]
            self.valid_starts = np.where(start_segment == end_segment)[0].astype(np.int64)
            if len(self.valid_starts) == 0:
                raise ValueError(
                    "No valid sequences remain after applying segment boundaries. "
                    "Decrease sequence_length or check segment_ids."
                )

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.valid_starts):
            raise IndexError("Index out of range for available sequences")
        start = int(self.valid_starts[index])
        end = start + self.sequence_length
        X = torch.from_numpy(self.input_data[start:end]).clone()
        y = torch.from_numpy(self.skeleton_data[end - 1]).clone()
        return X, y


class PressureSkeletonSequenceDataset(PressureSkeletonDataset):
    """Window-to-window variant for sequence-to-sequence supervision.

    Returns:
        tuple (torch.Tensor, torch.Tensor):
            - Input tensor with shape sequence_len x feature_dim
            - Target tensor with shape sequence_len x feature_dim_skeleton
    """

    def __getitem__(self, index):
        if index < 0 or index >= len(self.valid_starts):
            raise IndexError("Index out of range for available sequences")
        start = int(self.valid_starts[index])
        end = start + self.sequence_length
        X = torch.from_numpy(self.input_data[start:end]).clone()
        y = torch.from_numpy(self.skeleton_data[start:end]).clone()
        return X, y
    

class PressureDataset(Dataset):
    """
    PyTorch custom dataset for prediction input data.
    
    Args:
        features (np.ndarray): Input data to run prediction on.
    """
    def __init__(self, features):
        # Convert data to float PyTorch tensor
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        # Dataset length equals number of rows
        return len(self.features)

    def __getitem__(self, idx):
        # Return a single row of data
        return self.features[idx]