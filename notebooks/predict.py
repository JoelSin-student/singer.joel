# Prediction processor
import argparse
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler

from notebooks.loader import (
    calculate_grad,
    get_datapath_pairs,
    load_and_combine_data,
    load_config,
    restructure_insole_data,
)
from notebooks.model import SoleFormer, Transformer_Encoder, Transformer_Encoder_Seq2Seq, save_predictions


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def _build_input_tag(tags):
    unique_tags = []
    for tag in tags:
        tag_str = str(tag).strip()
        if tag_str and tag_str not in unique_tags:
            unique_tags.append(tag_str)
    if not unique_tags:
        return "unknown"
    return "__".join(unique_tags)


def infer_model_config_from_checkpoint(checkpoint, fallback_num_joints):
    model_config = dict(checkpoint.get("model_config", {}))
    state_dict = checkpoint["model_state_dict"]
    decoder_out_dim = None
    first_linear_weight = None
    decoder_weight = None

    if "output_decoder.4.weight" in state_dict:
        first_linear_weight = state_dict["feature_extractor.0.weight"]
        decoder_weight = state_dict["output_decoder.4.weight"]
        inferred_mode = "simple_seq2seq"
        decoder_out_dim = decoder_weight.shape[0]
    elif "output_decoder_l2.2.weight" in state_dict:
        first_linear_weight = state_dict["feature_extractor.0.weight"]
        decoder_weight = state_dict["output_decoder_l2.2.weight"]
        inferred_mode = "original"
        decoder_out_dim = decoder_weight.shape[0]
    elif any(k.startswith("fusion_decoder.") and k.endswith(".weight") for k in state_dict):
        inferred_mode = "soleformer"

        imu_weight = state_dict.get("imu_feature_extractor.0.weight", None)
        if imu_weight is not None:
            model_config.setdefault("imu_dim", imu_weight.shape[1])
            model_config.setdefault("d_model", imu_weight.shape[0])

        pressure_weight = state_dict.get("pressure_feature_extractor.0.weight", None)
        if pressure_weight is not None:
            model_config.setdefault("pressure_dim", pressure_weight.shape[1])
        else:
            # GraphPressureNet does not expose a direct input projection from pressure_dim.
            model_config.setdefault("pressure_dim", checkpoint.get("pressure_scaler_n_features", None))

        fusion_weight_keys = [
            k for k in state_dict if k.startswith("fusion_decoder.") and k.endswith(".weight")
        ]
        fusion_weight_keys.sort(key=lambda key: int(key.split(".")[1]))
        decoder_out_dim = state_dict[fusion_weight_keys[-1]].shape[0]
        model_config.setdefault("output_dim", decoder_out_dim)
        if model_config.get("pressure_dim") is not None and model_config.get("imu_dim") is not None:
            model_config.setdefault("input_dim", int(model_config["pressure_dim"]) + int(model_config["imu_dim"]))
    else:
        if model_config.get("model_mode") in {"original", "simple_seq2seq", "soleformer"}:
            inferred_mode = str(model_config["model_mode"]).lower()
            decoder_out_dim = int(model_config.get("output_dim", fallback_num_joints * 3))
        else:
            raise KeyError("Unable to infer decoder head from checkpoint state_dict.")

    if inferred_mode in {"original", "simple_seq2seq"}:
        if first_linear_weight is not None:
            model_config.setdefault("input_dim", first_linear_weight.shape[1])
            model_config.setdefault("d_model", first_linear_weight.shape[0])
        if decoder_weight is not None:
            model_config.setdefault("output_dim", decoder_weight.shape[0])
        if "input_dim" not in model_config or "d_model" not in model_config or "output_dim" not in model_config:
            raise KeyError(
                "Checkpoint is missing required model_config keys for mode "
                f"'{inferred_mode}'. Expected input_dim, d_model, and output_dim."
            )

    output_dim_for_joints = int(model_config.get("output_dim", decoder_out_dim or fallback_num_joints * 3))
    model_config.setdefault("num_joints", output_dim_for_joints // 3 or fallback_num_joints)
    model_config.setdefault("model_mode", checkpoint.get("model_mode", inferred_mode))

    if "num_encoder_layers" not in model_config:
        if inferred_mode in {"original", "simple_seq2seq"}:
            layer_ids = {
                int(key.split(".")[2])
                for key in state_dict
                if key.startswith("transformer_encoder.layers.")
            }
            model_config["num_encoder_layers"] = len(layer_ids)
        elif inferred_mode == "soleformer":
            layer_ids = {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("pressure_self_layers.") and ".in_proj_weight" in key
            }
            model_config["num_encoder_layers"] = len(layer_ids)

    return model_config


def load_minmax_scaler_from_checkpoint(checkpoint, prefix):
    required_keys = [
        f"{prefix}_scaler_min",
        f"{prefix}_scaler_scale",
        f"{prefix}_scaler_data_min",
        f"{prefix}_scaler_data_max",
        f"{prefix}_scaler_data_range",
        f"{prefix}_scaler_n_features",
    ]
    if not all(key in checkpoint for key in required_keys):
        return None

    scaler = MinMaxScaler()
    scaler.min_ = np.asarray(checkpoint[f"{prefix}_scaler_min"], dtype=np.float32)
    scaler.scale_ = np.asarray(checkpoint[f"{prefix}_scaler_scale"], dtype=np.float32)
    scaler.data_min_ = np.asarray(checkpoint[f"{prefix}_scaler_data_min"], dtype=np.float32)
    scaler.data_max_ = np.asarray(checkpoint[f"{prefix}_scaler_data_max"], dtype=np.float32)
    scaler.data_range_ = np.asarray(checkpoint[f"{prefix}_scaler_data_range"], dtype=np.float32)
    scaler.n_features_in_ = int(checkpoint[f"{prefix}_scaler_n_features"])
    scaler.n_samples_seen_ = 1
    return scaler


def start(args):
    config = load_config(args, args.config, args.model)

    model_mode = str(config["predict"].get("model_mode", "simple_seq2seq")).lower()
    if model_mode not in {"original", "simple_seq2seq", "soleformer"}:
        raise ValueError("predict.model_mode must be one of: original, simple_seq2seq, soleformer")

    skeleton_dir = os.path.join(config["location"]["data_path"], "skeleton")
    insole_dir = os.path.join(config["location"]["data_path"], "Insole")

    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)
    pair_tags = list(skeleton_insole_datapath_pairs.keys())
    input_tag = _build_input_tag(pair_tags)

    skeleton_df, insole_df, segment_ids = load_and_combine_data(skeleton_insole_datapath_pairs)
    pressure_lr_df, imu_lr_df, time_feature_df = restructure_insole_data(insole_df)

    sigma = float(config["predict"].get("smoothing_sigma", 0.0))
    if sigma > 0:
        pressure_lr_df = pressure_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        imu_lr_df = imu_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        print(f"Applied Gaussian smoothing before scaling (sigma={sigma}).")
    else:
        print("Gaussian smoothing disabled before scaling (smoothing_sigma=0).")

    checkpoint_path = config["predict"]["checkpoint_file"]
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model_config = infer_model_config_from_checkpoint(
        checkpoint,
        fallback_num_joints=skeleton_df.shape[1] // 3,
    )
    checkpoint_mode = str(checkpoint_model_config.get("model_mode", model_mode)).lower()
    if checkpoint_mode != model_mode:
        print(
            f"Warning: predict.model_mode={model_mode} but checkpoint indicates {checkpoint_mode}. "
            f"Using checkpoint mode {checkpoint_mode}."
        )
        model_mode = checkpoint_mode

    pressure_scaler = load_minmax_scaler_from_checkpoint(checkpoint, "pressure")
    imu_scaler = load_minmax_scaler_from_checkpoint(checkpoint, "imu")
    if pressure_scaler is None or imu_scaler is None:
        print("Warning: checkpoint is missing feature scalers; using fit_transform on prediction data.")
        pressure_scaler = MinMaxScaler()
        imu_scaler = MinMaxScaler()
        pressure_scaled = pressure_scaler.fit_transform(pressure_lr_df)
        imu_scaled = imu_scaler.fit_transform(imu_lr_df)
    else:
        pressure_scaled = pressure_scaler.transform(pressure_lr_df)
        imu_scaled = imu_scaler.transform(imu_lr_df)

    use_time_feature = _to_bool(
        checkpoint.get("preprocessing_use_time_feature", config["predict"].get("use_time_feature", False)),
        default=False,
    )
    if use_time_feature:
        # Normalize time feature per segment to avoid encoding global recording offsets.
        from notebooks.loader import normalize_time_feature_per_segment
        time_scaled = normalize_time_feature_per_segment(time_feature_df, segment_ids)
    else:
        time_scaled = None

    use_gradient_data = _to_bool(
        checkpoint.get("preprocessing_use_gradient_data", config["predict"].get("use_gradient_data", False)),
        default=False,
    )
    if use_gradient_data:
        grad_window_length = int(
            checkpoint.get("preprocessing_grad_window_length", config["predict"].get("grad_window_length", 5))
        )
        grad_polyorder = int(
            checkpoint.get("preprocessing_grad_polyorder", config["predict"].get("grad_polyorder", 2))
        )
        grad_smooth_grad1 = _to_bool(
            checkpoint.get("preprocessing_grad_smooth_grad1", config["predict"].get("grad_smooth_grad1", False)),
            default=False,
        )
        grad_feature_stats = None
        if all(key in checkpoint for key in ["grad_pressure_mean", "grad_pressure_std", "grad_imu_mean", "grad_imu_std"]):
            grad_feature_stats = {
                "pressure_mean": np.asarray(checkpoint["grad_pressure_mean"], dtype=np.float32),
                "pressure_std": np.asarray(checkpoint["grad_pressure_std"], dtype=np.float32),
                "imu_mean": np.asarray(checkpoint["grad_imu_mean"], dtype=np.float32),
                "imu_std": np.asarray(checkpoint["grad_imu_std"], dtype=np.float32),
            }

        pressure_scaled, imu_scaled = calculate_grad(
            pressure_scaled,
            imu_scaled,
            window_length=grad_window_length,
            polyorder=grad_polyorder,
            smooth_grad1=grad_smooth_grad1,
            normalization_stats=grad_feature_stats,
        )
        print(
            f"Derivative features enabled for prediction: input dim is "
            f"{pressure_scaled.shape[1] + imu_scaled.shape[1]}."
        )

    base_input_dim = pressure_scaled.shape[1] + imu_scaled.shape[1]
    input_feature_parts = [pressure_scaled, imu_scaled]
    if use_time_feature:
        input_feature_parts.append(time_scaled)
        print(
            f"Time feature enabled for prediction: input dim is "
            f"{base_input_dim + 1}."
        )

    input_feature_np = np.concatenate(input_feature_parts, axis=1)
    expected_input_dim = int(checkpoint_model_config.get("input_dim", input_feature_np.shape[1]))
    if expected_input_dim != input_feature_np.shape[1]:
        raise ValueError(
            f"Checkpoint expects input_dim={expected_input_dim}, but input pipeline produced "
            f"{input_feature_np.shape[1]}. Check use_gradient_data and preprocessing settings."
        )

    parameters = {
        "d_model": checkpoint_model_config.get("d_model", config["predict"]["d_model"]),
        "n_head": checkpoint_model_config.get("nhead", config["predict"]["n_head"]),
        "num_encoder_layer": checkpoint_model_config.get("num_encoder_layers", config["predict"]["num_encoder_layer"]),
        "dropout": config["predict"]["dropout"],
        "sequence_len": config["predict"]["sequence_len"],
        "input_dim": input_feature_np.shape[1],
        "num_joints": checkpoint_model_config.get("num_joints", skeleton_df.shape[1] // 3),
        "output_dim": checkpoint_model_config.get("output_dim", skeleton_df.shape[1]),
        "pressure_dim": checkpoint_model_config.get("pressure_dim", pressure_scaled.shape[1]),
        "imu_dim": checkpoint_model_config.get("imu_dim", imu_scaled.shape[1]),
        "num_dims": 3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_mode == "simple_seq2seq":
        model = Transformer_Encoder_Seq2Seq(
            input_dim=parameters["input_dim"],
            d_model=parameters["d_model"],
            nhead=parameters["n_head"],
            num_encoder_layers=parameters["num_encoder_layer"],
            num_joints=parameters["num_joints"],
            num_dims=parameters["num_dims"],
            dropout=parameters["dropout"],
        ).to(device)
    elif model_mode == "soleformer":
        model = SoleFormer(
            pressure_dim=parameters["pressure_dim"],
            imu_dim=parameters["imu_dim"],
            d_model=parameters["d_model"],
            nhead=parameters["n_head"],
            num_encoder_layers=parameters["num_encoder_layer"],
            output_dim=parameters["output_dim"],
            dropout=parameters["dropout"],
        ).to(device)
    else:
        model = Transformer_Encoder(
            input_dim=parameters["input_dim"],
            d_model=parameters["d_model"],
            nhead=parameters["n_head"],
            num_encoder_layers=parameters["num_encoder_layer"],
            num_joints=parameters["num_joints"],
            num_dims=parameters["num_dims"],
            dropout=parameters["dropout"],
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    input_tensor = torch.tensor(input_feature_np, dtype=torch.float32).to(device)

    max_start = len(input_tensor) - parameters["sequence_len"] + 1
    if max_start <= 0:
        raise ValueError(
            f"Input length ({len(input_tensor)}) is shorter than sequence_len ({parameters['sequence_len']})."
        )

    start_segment = np.asarray(segment_ids[:max_start])
    end_segment = np.asarray(segment_ids[parameters["sequence_len"] - 1 :])
    valid_starts = np.where(start_segment == end_segment)[0].astype(np.int64)
    if len(valid_starts) == 0:
        raise ValueError("No valid prediction windows remain after segment boundary filtering.")

    max_windows = config["predict"].get("max_windows", None)
    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows > 0:
            valid_starts = valid_starts[:max_windows]
            print(f"Limiting rollout to first {len(valid_starts)} windows (max_windows={max_windows}).")

    model.eval()
    all_predictions = []
    output_frame_indices = []

    with torch.no_grad():
        for start_idx in valid_starts:
            sequence = input_tensor[start_idx : start_idx + parameters["sequence_len"]].unsqueeze(0)
            prediction_raw = model(sequence)
            if prediction_raw.ndim == 3:
                prediction = prediction_raw[:, -1, :]
            else:
                prediction = prediction_raw
            all_predictions.append(prediction.detach().cpu().clone())
            output_frame_indices.append(start_idx + parameters["sequence_len"] - 1)

    final_predictions = torch.cat(all_predictions, dim=0).numpy()

    if "skeleton_scaler_mean" in checkpoint and "skeleton_scaler_scale" in checkpoint:
        skel_mean = np.asarray(checkpoint["skeleton_scaler_mean"], dtype=np.float32)
        skel_scale = np.asarray(checkpoint["skeleton_scaler_scale"], dtype=np.float32)
        final_predictions = final_predictions * skel_scale + skel_mean
        print("Applied skeleton inverse-transform (StandardScaler).")

    print(f"Prediction finished. Output shape: {final_predictions.shape}")
    target_column_names = checkpoint.get("target_column_names", None)
    save_predictions(
        final_predictions,
        args.model,
        frame_indices=output_frame_indices,
        output_stem=f"{input_tag}_{model_mode}",
        column_names=target_column_names,
    )


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description="Prediction Processor")

    parser.add_argument("--model", choices=["transformer_encoder", "transformer", "BERT"], default="transformer_encoder", help="Model selection")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML file")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--checkpoint_file", type=str, default=None)

    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--num_encoder_layer", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sequence_len", type=int, default=None)

    parser.add_argument("--smoothing_sigma", type=float, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--model_mode", type=str, default=None, choices=["original", "simple_seq2seq", "soleformer"])
    parser.add_argument("--use_time_feature", type=str, default=None)
    parser.add_argument("--use_gradient_data", type=str, default=None)
    parser.add_argument("--grad_window_length", type=int, default=None)
    parser.add_argument("--grad_polyorder", type=int, default=None)
    parser.add_argument("--grad_smooth_grad1", type=str, default=None)

    return parser
