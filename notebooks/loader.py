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

def load_config(args,config_path, model):
    """Load a YAML file from the given path and return it as a Python object.
    Args:
        path (str): File path of the YAML configuration file to load.
    Returns:
        dict: Dictionary converted from YAML content.
    """
    # If config_path is not provided, infer it from model name and mode.
    if config_path is None:
        config_path = f'config/{model}/{args.mode}.yaml'
    
    # Print the config file currently in use.
    print(f"<load config>")
    print(f"Loading configuration from: {config_path}")
    time.sleep(0.5)

    # Load config file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config values with command-line arguments
    cli_args = vars(args)   # Convert args Namespace into a dictionary
    for key, value in cli_args.items():  # Only override when the CLI value is not None
        if value is not None:           #
            config[key] = value         #

    return config


def get_datapath_pairs(skeleton_dir, insole_dir):
    """Pair skeleton/insole file paths sharing the same tag from directories.
    Args:
        skeleton_dir (str): Directory containing skeleton files (*_skeleton.csv).
        insole_dir (str): Directory containing insole files (*_Insole_*.csv).
    Returns:
        defaultdict: Dictionary keyed by tag.
            Value format: `{tag:{'skeleton': str, 'insole': list[str]}}`
    """
    # Show data paths
    print("---"*20)
    print(f"<Dataset Infomation>")
    print(f"skeleton data path : {skeleton_dir}")
    print(f"Inosole data path : {insole_dir}")
    time.sleep(0.5)

    # Get all CSV files in each folder
    skeleton_files = glob.glob(os.path.join(skeleton_dir, "*_skeleton.csv"))
    insole_files = glob.glob(os.path.join(insole_dir, "*_Insole_*.csv"))

    # Create dictionary to store skeleton/insole data pairs
    data_pairs = defaultdict(lambda: {'skeleton': None, 'insole': []})

    # Extract tags from skeleton files and store them
    for file_path in skeleton_files:
        filename = os.path.basename(file_path)
        tag = filename.replace('_skeleton.csv', '')
        data_pairs[tag]['skeleton'] = file_path
    
    # Extract tags from insole files and store them
    for file_path in insole_files:
        filename = os.path.basename(file_path)
        tag = filename.split('_Insole_')[0]

        # Add only when the matching skeleton file exists
        if tag in data_pairs:
            data_pairs[tag]['insole'].append(file_path)

    # Print extracted pairing results
    data_i=0
    for tag, paths in data_pairs.items():
        data_i+=1
        print(f"")
        print(f"Data_{data_i}_{tag}")
        print(f"skeleton: {paths['skeleton']}")
        print(*[f"insole: {f}" for f in sorted(paths['insole'])], sep='\n')
        time.sleep(0.3)
    print("---"*20)

    return data_pairs


def load_and_combine_data(data_pairs):
    """Load data from file-path dict and return concatenated DataFrames.
    Args:
        data_pairs (dict): Object with tags as keys and file-path dictionaries as values.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of concatenated skeleton and insole DataFrames.
    """
    # Create lists to store each data category
    all_skeleton_df = []
    all_insole_df   = []
    
    # Load each file as DataFrame and append to lists
    for tag, paths in data_pairs.items():
        skeleton_df     = pd.read_csv(paths['skeleton'])
        insole_df  = pd.read_csv(paths['insole'])

        all_skeleton_df.append(skeleton_df)
        all_insole_df.append(insole_df)

    return (pd.concat(all_skeleton_df, ignore_index=True),
            pd.concat(all_insole_df, ignore_index=True))


def restructure_insole_data(insole_df):
    """Split insole data into pressure/IMU.
    Args:
        insole_df (pd.DataFrame): DataFrame containing insole data.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of split/re-combined DataFrames.
    """
    # Extract sensor groups from left-foot data
    pressure_lr = insole_df.drop(["left acceleration X[g]","left acceleration Y[g]","left acceleration Z[g]",
                                  "left angular X[dps]","left angular Y[dps]","left angular Z[dps]",
                                  "left total force[N]","left center of pressure X[-0.5...+0.5]","left center of pressure Y[-0.5...+0.5]",
                                  "right acceleration X[g]","right acceleration Y[g]","right acceleration Z[g]",
                                  "right angular X[dps]","right angular Y[dps]","right angular Z[dps]",
                                  "right total force[N]","right center of pressure X[-0.5...+0.5]","right center of pressure Y[-0.5...+0.5]",
                                  "right steps[]","left steps[]"],axis=1)
    IMU_lr      = insole_df[["left acceleration X[g]","left acceleration Y[g]","left acceleration Z[g]",
                             "left angular X[dps]","left angular Y[dps]","left angular Z[dps]",
                             "right acceleration X[g]","right acceleration Y[g]","right acceleration Z[g]",
                             "right angular X[dps]","right angular Y[dps]","right angular Z[dps]","right steps[]","left steps[]"]]

    return pressure_lr, IMU_lr


def calculate_grad(pressure_lr, IMU_lr):
    """Compute 1st/2nd derivatives for pressure and IMU, then concatenate.
    Args:
        pressure_lr (np): Pressure-sensor time-series data.
        IMU_lr (np): IMU time-series data.
    Returns:
        tuple[np, np]: Tuple containing expanded feature arrays.
    """
    # First/second derivative features (optional)
    pressure_grad1 = np.gradient(pressure_lr, axis=0)
    pressure_grad2 = np.gradient(pressure_grad1, axis=0)
    IMU_grad1 = np.gradient(IMU_lr, axis=0)
    IMU_grad2 = np.gradient(IMU_grad1, axis=0)
    pressure_features = np.concatenate([
        pressure_lr,
        pressure_grad1,
        pressure_grad2,
    ], axis=1)
    IMU_features = np.concatenate([
        IMU_lr,
        IMU_grad1,
        IMU_grad2,
    ], axis=1)

    return pressure_features, IMU_features
    

# # Original PressureSkeletonDataset
# # Disabled for debugging work
# class PressureSkeletonDataset(Dataset):
#     """PyTorch custom dataset for pressure and skeleton data.
#     Args:
#         pressure_data (pd): Pressure-data sequence.
#         skeleton_data (pd): Skeleton-data sequence.

#     Returns:
#         pressure_data (torch.Tensor): Pressure data converted to Tensor.
#         skeleton_data (torch.Tensor): Skeleton data converted to Tensor.
#     """
#     def __init__(self, input_feature, skeleton_data, sequence_length):
#         self.sequence_length = sequence_length
#         self.input_data = input_feature
#         self.skeleton_data = skeleton_data
        
#     def __len__(self):
#         return len(self.input_data) - self.sequence_length + 1
    
#     def __getitem__(self, index):
#         # Slice input sequence
#         X = self.input_data[index : index + self.sequence_length]
#         y = self.skeleton_data[index + self.sequence_length - 1]

#         return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# Debug-friendly PressureSkeletonDataset
class PressureSkeletonDataset(Dataset):
    """PyTorch custom dataset for pressure and skeleton data.

    Args:
        input_feature (array-like): Sequence of pressure and IMU features.
        skeleton_data (array-like): Sequence of corresponding skeleton data.
        sequence_length (int): Number of frames fed to the model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Input tensor with shape sequence_len x feature_dim
            - Target tensor with shape feature_dim_skeleton
    """

    def __init__(self, input_feature, skeleton_data, sequence_length):
        self.sequence_length = int(sequence_length)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")

        self.input_data = np.asarray(input_feature, dtype=np.float32)
        self.skeleton_data = np.asarray(skeleton_data, dtype=np.float32)

        if len(self.input_data) != len(self.skeleton_data):
            raise ValueError(
                "input_feature and skeleton_data must contain the same number of frames"
            )

        self._valid_length = len(self.input_data) - self.sequence_length + 1
        if self._valid_length <= 0:
            raise ValueError(
                "Not enough frames to build a sequence. Increase data length or decrease sequence_length"
            )

    def __len__(self):
        return self._valid_length

    def __getitem__(self, index):
        if index < 0 or index >= self._valid_length:
            raise IndexError("Index out of range for available sequences")
        start = index
        end = start + self.sequence_length
        X = torch.from_numpy(self.input_data[start:end]).clone()
        y = torch.from_numpy(self.skeleton_data[end - 1]).clone()
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