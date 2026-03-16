# Prediction processor
#
#
#
#
import numpy as np
import argparse
import torch
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from notebooks.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, calculate_grad, load_config, PressureDataset
from notebooks.model import Transformer_Encoder, save_predictions


def infer_model_config_from_checkpoint(checkpoint, fallback_num_joints):
    model_config = dict(checkpoint.get("model_config", {}))
    state_dict = checkpoint["model_state_dict"]

    first_linear_weight = state_dict["feature_extractor.0.weight"]
    decoder_weight = state_dict["output_decoder.6.weight"]

    model_config.setdefault("input_dim", first_linear_weight.shape[1])
    model_config.setdefault("d_model", first_linear_weight.shape[0])
    model_config.setdefault("num_joints", decoder_weight.shape[0] // 3 or fallback_num_joints)

    if "num_encoder_layers" not in model_config:
        layer_ids = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("transformer_encoder.layers.")
        }
        model_config["num_encoder_layers"] = len(layer_ids)

    return model_config


def summarize_variation(name, array_2d):
    array_2d = np.asarray(array_2d)
    if array_2d.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={array_2d.shape}")

    col_std_mean = float(np.std(array_2d, axis=0).mean())
    row_std_mean = float(np.std(array_2d, axis=1).mean())

    if len(array_2d) > 1:
        frame_delta_l2 = np.linalg.norm(np.diff(array_2d, axis=0), axis=1)
        delta_l2_mean = float(frame_delta_l2.mean())
        delta_l2_min = float(frame_delta_l2.min())
        delta_zero_ratio = float((frame_delta_l2 < 1e-12).mean())
    else:
        delta_l2_mean = 0.0
        delta_l2_min = 0.0
        delta_zero_ratio = 1.0

    print(
        f"{name}: col_std_mean={col_std_mean:.6f}, row_std_mean={row_std_mean:.6f}, "
        f"delta_l2_mean={delta_l2_mean:.6f}, delta_l2_min={delta_l2_min:.6f}, "
        f"delta_zero_ratio={delta_zero_ratio:.6f}"
    )

    return {
        "col_std_mean": col_std_mean,
        "row_std_mean": row_std_mean,
        "delta_l2_mean": delta_l2_mean,
        "delta_l2_min": delta_l2_min,
        "delta_zero_ratio": delta_zero_ratio,
    }


def probe_model_sensitivity(model, input_tensor, sequence_len):
    total_frames = int(len(input_tensor))
    if total_frames < sequence_len:
        print("Model sensitivity probe skipped: not enough frames for one sequence.")
        return

    start_a = 0
    start_b = max(0, (total_frames - sequence_len) // 2)

    with torch.no_grad():
        seq_a = input_tensor[start_a:start_a + sequence_len].unsqueeze(0)
        seq_b = input_tensor[start_b:start_b + sequence_len].unsqueeze(0)
        out_a = model(seq_a)
        out_b = model(seq_b)
        real_diff_l2 = float(torch.norm(out_a - out_b).item())

        rand_a = torch.randn_like(seq_a)
        rand_b = torch.randn_like(seq_a)
        out_rand_a = model(rand_a)
        out_rand_b = model(rand_b)
        random_diff_l2 = float(torch.norm(out_rand_a - out_rand_b).item())

    print("\n<model sensitivity probe>")
    print(f"real_window_output_diff_l2={real_diff_l2:.10f}")
    print(f"random_window_output_diff_l2={random_diff_l2:.10f}")
    if real_diff_l2 < 1e-5 and random_diff_l2 < 1e-5:
        print(
            "Warning: checkpoint appears collapsed (near-constant output even for very different inputs)."
        )

def start(args):

    # Load YAML config
    config = load_config(args, args.config, args.model)

    # Set data paths
    skeleton_dir = config["location"]["data_path"] + "/skeleton/"
    insole_dir   = config["location"]["data_path"] + "/Insole/"
    
    # Preprocess skeleton and insole data
    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)     # Collect paired skeleton/insole file paths
    output_stem = None
    pair_tags = list(skeleton_insole_datapath_pairs.keys())
    if len(pair_tags) == 1:
        output_stem = pair_tags[0]
    elif len(pair_tags) > 1:
        print(
            "Multiple testing tags detected; using default output name "
            "'predicted_skeleton.csv'."
        )
    skeleton_df, insole_df  = load_and_combine_data(skeleton_insole_datapath_pairs)   # Load skeleton and insole data
    pressure_lr_df, IMU_lr_df = restructure_insole_data(insole_df)                    # Split pressure/IMU

    # Match training-time preprocessing: smooth sensor streams before scaling.
    sigma = float(config["predict"].get("smoothing_sigma", 2.0))
    if sigma > 0:
        pressure_lr_df = pressure_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        IMU_lr_df = IMU_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        print(f"Applied Gaussian smoothing before scaling (sigma={sigma}).")

    # Initialize scalers
    pressure_normalizer = MinMaxScaler()
    imu_normalizer      = MinMaxScaler()

    # Fit and transform with scaler
    pressure_scaled  =  pressure_normalizer.fit_transform(pressure_lr_df)   # Pressure data (fit + transform)
    IMU_scaled       =  imu_normalizer.fit_transform(IMU_lr_df)             # IMU data (fit + transform)

    print("\n<variation check: before/after scaling>")
    summarize_variation("pressure_raw", pressure_lr_df.to_numpy(dtype=np.float32))
    summarize_variation("imu_raw", IMU_lr_df.to_numpy(dtype=np.float32))
    summarize_variation("pressure_scaled", pressure_scaled)
    summarize_variation("imu_scaled", IMU_scaled)

    # Load checkpoint before building features so prediction matches the training-time input layout.
    checkpoint = torch.load(config["predict"]["checkpoint_file"], map_location="cpu")
    checkpoint_model_config = infer_model_config_from_checkpoint(
        checkpoint,
        fallback_num_joints=skeleton_df.shape[1] // 3
    )

    base_input_dim = pressure_scaled.shape[1] + IMU_scaled.shape[1]
    expected_input_dim = checkpoint_model_config.get("input_dim", base_input_dim)

    if expected_input_dim == base_input_dim * 3:
        pressure_scaled, IMU_scaled = calculate_grad(pressure_scaled, IMU_scaled)
        print("<variation check: with derivative-expanded features>")
        summarize_variation("pressure_grad_features", pressure_scaled)
        summarize_variation("imu_grad_features", IMU_scaled)
    elif expected_input_dim != base_input_dim:
        raise ValueError(
            f"Checkpoint expects input_dim={expected_input_dim}, but prediction pipeline produced "
            f"base_dim={base_input_dim}. Supported cases are raw features ({base_input_dim}) or "
            f"raw+derivatives ({base_input_dim * 3})."
        )

    input_feature_np = np.concatenate([pressure_scaled, IMU_scaled], axis=1)

    # Final parameter setup
    parameters = {
        # Model parameters
        "d_model"            : checkpoint_model_config.get("d_model", config["predict"]["d_model"]),
        "n_head"             : checkpoint_model_config.get("nhead", config["predict"]["n_head"]),
        "num_encoder_layer"  : checkpoint_model_config.get("num_encoder_layers", config["predict"]["num_encoder_layer"]),
        "dropout"            : config["predict"]["dropout"],
        "batch_size"         : config["predict"]["batch_size"],
        "sequence_len"       : config["predict"]["sequence_len"],

        # Other settings
        "input_dim"          : input_feature_np.shape[1],
        "num_joints"         : checkpoint_model_config.get("num_joints", skeleton_df.shape[1] // 3),
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

    # Load checkpoint weights
    checkpoint = torch.load(parameters["checkpoint_file"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # List to store predictions and corresponding original frame indices
    all_predictions = []
    output_frame_indices = []

    # Convert input data to tensor
    input_tensor = torch.tensor(input_feature_np, dtype=torch.float32).to(device)

    # Check whether loaded checkpoint is input-sensitive before full rollout.
    probe_model_sensitivity(model, input_tensor, parameters["sequence_len"])

    model.eval()
    with torch.no_grad():
        num_windows = len(input_tensor) - parameters["sequence_len"] + 1
        if num_windows <= 0:
            raise ValueError(
                f"Input length ({len(input_tensor)}) is shorter than sequence_len "
                f"({parameters['sequence_len']})."
            )

        for i in range(num_windows):  # Iterate sequence windows
            sequence = input_tensor[i : i + parameters["sequence_len"]]  # Slice by sequence length
            sequence = sequence.unsqueeze(0)                             # Convert to [1, sequence_len, features]
            prediction = model(sequence)                                 # Run model prediction
            all_predictions.append(prediction.detach().cpu().clone())    # Store prediction
            output_frame_indices.append(i + parameters["sequence_len"] - 1)

    # Merge all predictions into a single tensor
    final_predictions = torch.cat(all_predictions, dim=0)
    final_predictions_np = final_predictions.numpy()

    # Quick diagnostics for debugging nearly-constant outputs.
    print("\n<variation check: model input/output>")
    input_stats = summarize_variation("model_input", input_feature_np)
    output_stats = summarize_variation("model_output", final_predictions_np)
    if output_stats["col_std_mean"] < 1e-6:
        print(
            "Warning: predictions are nearly constant across frames. "
            "This often indicates a model/checkpoint or feature mismatch issue."
        )

    # Save per-frame debug traces to pinpoint where variation disappears.
    os.makedirs("./results/output", exist_ok=True)
    input_debug = pd.DataFrame({
        "Frame": np.arange(len(input_feature_np), dtype=np.int64),
        "pressure_row_std": np.std(np.asarray(pressure_scaled), axis=1),
        "imu_row_std": np.std(np.asarray(IMU_scaled), axis=1),
        "model_input_row_std": np.std(input_feature_np, axis=1),
    })
    input_delta = np.linalg.norm(np.diff(input_feature_np, axis=0), axis=1) if len(input_feature_np) > 1 else np.array([], dtype=np.float32)
    input_debug["model_input_delta_l2_prev"] = np.concatenate([[np.nan], input_delta])
    input_debug_path = "./results/output/predict_input_debug.csv"
    input_debug.to_csv(input_debug_path, index=False)

    output_debug = pd.DataFrame({
        "Frame": np.asarray(output_frame_indices, dtype=np.int64),
        "model_output_row_std": np.std(final_predictions_np, axis=1),
    })
    output_delta = np.linalg.norm(np.diff(final_predictions_np, axis=0), axis=1) if len(final_predictions_np) > 1 else np.array([], dtype=np.float32)
    output_debug["model_output_delta_l2_prev"] = np.concatenate([[np.nan], output_delta])
    output_debug_path = "./results/output/predict_output_debug.csv"
    output_debug.to_csv(output_debug_path, index=False)

    print(f"Saved input debug trace to {input_debug_path}")
    print(f"Saved output debug trace to {output_debug_path}")

    print(f"Prediction finished. Output shape: {final_predictions_np.shape}")
    save_predictions(
        final_predictions_np,
        args.model,
        frame_indices=output_frame_indices,
        output_stem=output_stem,
    )
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