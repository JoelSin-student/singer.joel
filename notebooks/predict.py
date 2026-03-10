# Prediction processor
#
#
#
#
import numpy as np
import pandas as pd
import argparse
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from notebooks.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, load_config, PressureDataset
from notebooks.model import Transformer_Encoder, save_predictions

def start(args):

    # Load YAML config
    config = load_config(args, args.config, args.model)

    # Set data paths
    skeleton_dir = config["location"]["data_path"] + "/skeleton/"
    insole_dir   = config["location"]["data_path"] + "/Insole/"
    
    # Preprocess skeleton and insole data
    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)     # Collect paired skeleton/insole file paths
    skeleton_df, insole_df  = load_and_combine_data(skeleton_insole_datapath_pairs)   # Load skeleton, right-insole, and left-insole data
    pressure_lr_df, IMU_lr_df = restructure_insole_data(insole_df)                    # Split pressure/IMU and merge left+right

    # Initialize scalers
    pressure_normalizer = MinMaxScaler()
    imu_normalizer = MinMaxScaler()

    # Fit and transform with scaler
    pressure_scaled  =  pressure_normalizer.fit_transform(pressure_lr_df)   # Pressure data (fit + transform)
    IMU_scaled       =  imu_normalizer.fit_transform(IMU_lr_df)             # IMU data (fit + transform)
    input_feature_np = np.concatenate([pressure_scaled, IMU_scaled], axis=1)

    # Final parameter setup
    parameters = {
        # Model parameters
        "d_model"            : config["predict"]["d_model"],
        "n_head"             : config["predict"]["n_head"],
        "num_encoder_layer"  : config["predict"]["num_encoder_layer"],
        "dropout"            : config["predict"]["dropout"],
        "batch_size"         : config["predict"]["batch_size"],
        "sequence_len"       : config["predict"]["sequence_len"],

        # Other settings
        "input_dim"          : pressure_lr_df.shape[1] + IMU_lr_df.shape[1], # Total dims of pressure + gyro + acceleration
        "num_joints"         : skeleton_df.shape[1] // 3,  # Divide by 3 for 3D coordinates
        "num_dims"           :  3,
        "checkpoint_file"    : config["predict"]["checkpoint_file"]
    }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Initialize model (using fixed parameters)
    model = Transformer_Encoder(
        input_dim=parameters["input_dim"], 
        d_model= parameters["d_model"],
        nhead=parameters["n_head"],
        num_encoder_layers=parameters["num_encoder_layer"],
        num_joints=parameters["num_joints"],
        num_dims=parameters["num_dims"],
        dropout=parameters["dropout"]
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(parameters["checkpoint_file"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # List to store predictions
    all_predictions = []

    # Convert input data to tensor
    input_tensor = torch.tensor(input_feature_np, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(len(input_tensor) - parameters["sequence_len"]):  # Iterate sequence windows
            sequence = input_tensor[i : i + parameters["sequence_len"]]  # Slice by sequence length
            sequence = sequence.unsqueeze(0)                             # Convert to [1, sequence_len, features]
            prediction = model(sequence)                                 # Run model prediction
            all_predictions.append(prediction.detach().cpu().clone())    # Store prediction

    # Merge all predictions into a single tensor
    final_predictions = torch.cat(all_predictions, dim=0)
    final_predictions_np = final_predictions.numpy()
    # skeleton_scaler = joblib.load('./scaler/skeleton_scaler.pkl')
    # final_predictions_np = skeleton_scaler.inverse_transform(final_predictions_np)

    print(f"Prediction finished. Output shape: {final_predictions_np.shape}")
    save_predictions(final_predictions_np, args.model)
    return 


@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Training Processor')

    # Basic settings
    parser.add_argument('--model', choices=['transformer_encoder','transformer', 'BERT'], default='transformer_encoder', help='Model selection')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML file')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--checkpoint_file', type=str, default=None)
    parser.add_argument('--sequence_len', type=int, default=None)

    # Model parameters
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--num_encoder_layer', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    return parser