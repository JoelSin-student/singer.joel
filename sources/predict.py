# Prediction processor
import argparse
import os
import re
import time
from typing import Any, cast
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler

from sources.loader import (
    calculate_grad,
    get_datapath_pairs,
    load_and_combine_data,
    load_config,
    restructure_insole_data,
)
from sources.model import SoleFormer, Transformer_Encoder, Transformer_Encoder_Seq2Seq, save_predictions
from sources.util import format_ablation_tag, join_nonempty, resolve_ablation_id


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


def _next_prediction_cycle_id(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^Predicted_skeleton_cycle_(\d+)_")
    max_id = 0
    for path in output_dir.glob("Predicted_skeleton_cycle_*.csv"):
        m = pattern.match(path.stem)
        if not m:
            continue
        max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def _split_predictions_by_segment(predictions, frame_indices, segment_ids, data_pairs):
    sorted_tags = [tag for tag, _ in sorted(data_pairs.items())]
    frame_idx = np.asarray(frame_indices, dtype=np.int64)
    output_segment_ids = np.asarray(segment_ids, dtype=np.int64)[frame_idx]

    split_outputs = []
    for seg_id in sorted(np.unique(output_segment_ids).tolist()):
        tag_index = int(seg_id) - 1
        if tag_index < 0 or tag_index >= len(sorted_tags):
            raise ValueError(f"Invalid segment id {seg_id} for {len(sorted_tags)} paired tags.")

        tag = sorted_tags[tag_index]
        mask = output_segment_ids == seg_id
        split_outputs.append(
            {
                "tag": tag,
                "predictions": predictions[mask],
                "frame_indices": frame_idx[mask],
            }
        )

    return split_outputs


def _extract_subject_key_from_tag(tag):
    tag_str = str(tag).strip()
    if not tag_str:
        return ""
    return tag_str.split("_", 1)[0]


def _build_segment_subject_map(data_pairs):
    sorted_tags = [tag for tag, _ in sorted(data_pairs.items())]
    segment_subject = {}
    for segment_id, tag in enumerate(sorted_tags, start=1):
        segment_subject[segment_id] = _extract_subject_key_from_tag(tag)
    return segment_subject


def _map_segments_to_subject_keys(segment_ids, segment_subject_map):
    subject_keys = []
    missing_segments = []
    for seg_id in np.asarray(segment_ids, dtype=np.int64):
        key = segment_subject_map.get(int(seg_id), "")
        if not key:
            missing_segments.append(int(seg_id))
        subject_keys.append(key)

    if missing_segments:
        missing_segments = sorted(set(missing_segments))
        raise KeyError(f"Missing subject mapping for segment id(s): {missing_segments}")

    return np.asarray(subject_keys, dtype=object)


def _load_subject_height_map():
    repo_root = Path(__file__).resolve().parents[1]
    subject_info_path = repo_root / "data" / "subject_info.txt"
    if not subject_info_path.is_file():
        raise FileNotFoundError(f"Subject info file not found: {subject_info_path}")

    subject_info = np.genfromtxt(subject_info_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    dtype_names = subject_info.dtype.names or ()
    if "subject_key" not in dtype_names or "height" not in dtype_names:
        raise ValueError(
            "subject_info.txt must include 'subject_key' and 'height' columns for height denormalization."
        )

    height_map = {}
    for row in np.atleast_1d(subject_info):
        key = str(row["subject_key"]).strip()
        if not key:
            continue
        height_map[key] = float(row["height"])
    return height_map


def _resolve_position_column_indices(target_column_names, target_position_columns, num_output_cols):
    if target_column_names is None:
        # Legacy checkpoints without column names: assume XYZ-only output.
        if num_output_cols % 3 == 0:
            return list(range(num_output_cols))
        return []

    columns = list(target_column_names)
    index_by_name = {name: idx for idx, name in enumerate(columns)}

    if target_position_columns:
        indices = [index_by_name[name] for name in target_position_columns if name in index_by_name]
        if indices:
            return sorted(set(indices))

    # Fallback for coordinate-style headers.
    coord_pattern = re.compile(r"^(?:pos::)?[XYZ]\.\d+$")
    return [idx for idx, name in enumerate(columns) if coord_pattern.match(str(name))]


def _coords_look_height_normalized(values, max_abs_threshold=0.02):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return False
    max_abs = float(np.nanmax(np.abs(arr[finite])))
    return max_abs <= float(max_abs_threshold)


def _apply_subject_height_denorm(
    predictions,
    frame_indices,
    segment_ids,
    data_pairs,
    target_column_names,
    target_position_columns,
):
    if predictions.size == 0:
        return predictions

    coord_indices = _resolve_position_column_indices(
        target_column_names=target_column_names,
        target_position_columns=target_position_columns,
        num_output_cols=predictions.shape[1],
    )
    if not coord_indices:
        print("No position-coordinate columns detected for height denormalization; skipping.")
        return predictions

    sorted_tags = [tag for tag, _ in sorted(data_pairs.items())]
    if not sorted_tags:
        return predictions

    frame_idx = np.asarray(frame_indices, dtype=np.int64)
    if frame_idx.size != predictions.shape[0]:
        raise ValueError(
            f"Length mismatch for denormalization: frame_indices={frame_idx.size}, predictions={predictions.shape[0]}"
        )

    output_segment_ids = np.asarray(segment_ids, dtype=np.int64)[frame_idx]
    height_map = _load_subject_height_map()

    denorm = predictions.copy()
    coord_idx = np.asarray(coord_indices, dtype=np.int64)
    missing_subjects = []
    applied_segments = 0
    skipped_segments = 0
    for seg_id in np.unique(output_segment_ids):
        tag_index = int(seg_id) - 1
        if tag_index < 0 or tag_index >= len(sorted_tags):
            raise ValueError(f"Invalid segment id {seg_id} for {len(sorted_tags)} paired tags.")

        tag = sorted_tags[tag_index]
        subject_key = _extract_subject_key_from_tag(tag)
        if subject_key not in height_map:
            missing_subjects.append(subject_key)
            continue

        subject_height = float(height_map[subject_key])
        row_mask = output_segment_ids == seg_id
        row_idx = np.where(row_mask)[0]
        segment_coords = denorm[np.ix_(row_idx, coord_idx)]
        if not _coords_look_height_normalized(segment_coords):
            skipped_segments += 1
            continue

        denorm[np.ix_(row_idx, coord_idx)] = segment_coords * subject_height
        applied_segments += 1

    if missing_subjects:
        missing_subjects = sorted(set([m for m in missing_subjects if m]))
        raise KeyError(
            f"Missing subject heights in subject_info.txt for subject_key(s): {missing_subjects}"
        )

    print(
        "Subject-specific height denormalization summary: "
        f"coord_cols={len(coord_indices)}, applied_segments={applied_segments}, "
        f"skipped_segments={skipped_segments}."
    )
    return denorm


def _compute_acceleration(pos, fps=60.0):
    if pos["X"].shape[0] < 3:
        return {
            "X": np.empty((0, pos["X"].shape[1]), dtype=np.float64),
            "Y": np.empty((0, pos["Y"].shape[1]), dtype=np.float64),
            "Z": np.empty((0, pos["Z"].shape[1]), dtype=np.float64),
            "labels": pos["labels"],
        }

    scale = float(fps) ** 2
    return {
        "X": (pos["X"][2:, :] - 2.0 * pos["X"][1:-1, :] + pos["X"][:-2, :]) * scale,
        "Y": (pos["Y"][2:, :] - 2.0 * pos["Y"][1:-1, :] + pos["Y"][:-2, :]) * scale,
        "Z": (pos["Z"][2:, :] - 2.0 * pos["Z"][1:-1, :] + pos["Z"][:-2, :]) * scale,
        "labels": pos["labels"],
    }


def _compute_mpjace(gt, pred, fps=60.0):
    gt_a = _compute_acceleration(gt, fps=fps)
    pred_a = _compute_acceleration(pred, fps=fps)
    if gt_a["X"].shape[0] == 0 or pred_a["X"].shape[0] == 0:
        empty = np.full((len(gt["labels"]),), np.nan, dtype=np.float64)
        return {"full": np.nan, "per_joint": empty}

    dx = pred_a["X"] - gt_a["X"]
    dy = pred_a["Y"] - gt_a["Y"]
    dz = pred_a["Z"] - gt_a["Z"]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return {
        "full": float(np.nanmean(dist)),
        "per_joint": np.nanmean(dist, axis=0),
    }


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

        pressure_is_graph = any(k.startswith("pressure_feature_extractor.sensor_projection") for k in state_dict)
        model_config.setdefault("use_graph_pressure", bool(pressure_is_graph))
        model_config.setdefault(
            "use_single_attention",
            not any(k.startswith("imu_to_pressure_cross_layers.") for k in state_dict),
        )

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


def _load_subject_zscore_stats_from_checkpoint(checkpoint, prefix):
    required_keys = [
        f"{prefix}_subject_zscore_subject_keys",
        f"{prefix}_subject_zscore_mean",
        f"{prefix}_subject_zscore_std",
        f"{prefix}_subject_zscore_global_mean",
        f"{prefix}_subject_zscore_global_std",
    ]
    if not all(key in checkpoint for key in required_keys):
        return None

    keys = [str(k) for k in checkpoint[f"{prefix}_subject_zscore_subject_keys"]]
    means = checkpoint[f"{prefix}_subject_zscore_mean"]
    stds = checkpoint[f"{prefix}_subject_zscore_std"]
    if len(keys) != len(means) or len(keys) != len(stds):
        raise ValueError(
            f"Invalid subject z-score checkpoint payload for '{prefix}': "
            "subject keys, means, and stds lengths must match."
        )

    subject_stats = {}
    for i, key in enumerate(keys):
        subject_stats[key] = {
            "mean": np.asarray(means[i], dtype=np.float32),
            "std": np.asarray(stds[i], dtype=np.float32),
        }

    return {
        "subject": subject_stats,
        "global_mean": np.asarray(checkpoint[f"{prefix}_subject_zscore_global_mean"], dtype=np.float32),
        "global_std": np.asarray(checkpoint[f"{prefix}_subject_zscore_global_std"], dtype=np.float32),
    }


def _transform_subject_channel_zscore(features, subject_keys, zscore_stats, eps=1e-8):
    x = np.asarray(features, dtype=np.float32)
    keys = np.asarray(subject_keys, dtype=object)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {x.shape}")
    if len(x) != len(keys):
        raise ValueError("features and subject_keys must have the same number of rows")

    subject_stats = zscore_stats["subject"]
    global_mean = np.asarray(zscore_stats["global_mean"], dtype=np.float32)
    global_std = np.asarray(zscore_stats["global_std"], dtype=np.float32)
    global_std = np.maximum(global_std, np.float32(eps))

    out = np.zeros_like(x, dtype=np.float32)
    unseen_subjects = []
    for subject_key in sorted(set([str(k) for k in keys])):
        mask = keys == subject_key
        stats = subject_stats.get(subject_key)
        if stats is None:
            unseen_subjects.append(subject_key)
            mean = global_mean
            std = global_std
        else:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            std = np.maximum(std, np.float32(eps))
        out[mask] = (x[mask] - mean) / std

    return out, unseen_subjects


def start(args):
    config = load_config(args, args.config, args.model)

    model_mode = str(config["predict"].get("model_mode", "simple_seq2seq")).lower()
    if model_mode not in {"original", "simple_seq2seq", "soleformer"}:
        raise ValueError("predict.model_mode must be one of: original, simple_seq2seq, soleformer")

    abl_id = resolve_ablation_id(config, "predict")
    abl_tag = format_ablation_tag(abl_id)

    skeleton_dir = os.path.join(config["location"]["data_path"], "skeleton")
    insole_dir = os.path.join(config["location"]["data_path"], "Insole")

    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)
    combined_data = cast(tuple[Any, Any, Any], load_and_combine_data(skeleton_insole_datapath_pairs))
    skeleton_df, insole_df, segment_ids = combined_data
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

    segment_subject_map = _build_segment_subject_map(skeleton_insole_datapath_pairs)
    subject_keys = _map_segments_to_subject_keys(segment_ids, segment_subject_map)

    pressure_zscore_stats = _load_subject_zscore_stats_from_checkpoint(checkpoint, "pressure")
    imu_zscore_stats = _load_subject_zscore_stats_from_checkpoint(checkpoint, "imu")
    if pressure_zscore_stats is not None and imu_zscore_stats is not None:
        pressure_scaled, pressure_unseen = _transform_subject_channel_zscore(
            pressure_lr_df.to_numpy(),
            subject_keys,
            pressure_zscore_stats,
        )
        imu_scaled, imu_unseen = _transform_subject_channel_zscore(
            imu_lr_df.to_numpy(),
            subject_keys,
            imu_zscore_stats,
        )

        unseen_subjects = sorted(set(pressure_unseen + imu_unseen))
        if unseen_subjects:
            print(
                "Warning: prediction data includes subject key(s) not present in training z-score stats; "
                "using global training z-score for those rows: "
                f"{unseen_subjects}"
            )
        else:
            print("Applied subject-wise per-channel z-score normalization from checkpoint.")
    else:
        pressure_scaler = load_minmax_scaler_from_checkpoint(checkpoint, "pressure")
        imu_scaler = load_minmax_scaler_from_checkpoint(checkpoint, "imu")
        if pressure_scaler is None or imu_scaler is None:
            print(
                "Warning: checkpoint is missing feature scalers; using fit_transform on prediction data. "
                "This is only a fallback and may degrade results."
            )
            pressure_scaler = MinMaxScaler()
            imu_scaler = MinMaxScaler()
            pressure_scaled = pressure_scaler.fit_transform(pressure_lr_df)
            imu_scaled = imu_scaler.fit_transform(imu_lr_df)
        else:
            print("Applied legacy MinMax feature scaling from checkpoint.")
            pressure_scaled = pressure_scaler.transform(pressure_lr_df)
            imu_scaled = imu_scaler.transform(imu_lr_df)

    use_time_feature = _to_bool(
        checkpoint.get("preprocessing_use_time_feature", config["predict"].get("use_time_feature", False)),
        default=False,
    )
    if use_time_feature:
        # Normalize time feature per segment to avoid encoding global recording offsets.
        from sources.loader import normalize_time_feature_per_segment
        time_scaled = normalize_time_feature_per_segment(time_feature_df, segment_ids)
    else:
        time_scaled = None

    use_gradient_data = _to_bool(
        checkpoint.get("preprocessing_use_gradient_data", config["predict"].get("use_gradient_data", False)),
        default=False,
    )
    if model_mode == "soleformer" and use_gradient_data:
        print("SoleFormer mode: forcing use_gradient_data=False (derivative feature expansion is disabled).")
        use_gradient_data = False
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

        grad_outputs = calculate_grad(
            pressure_scaled,
            imu_scaled,
            window_length=grad_window_length,
            polyorder=grad_polyorder,
            smooth_grad1=grad_smooth_grad1,
            normalization_stats=grad_feature_stats,
        )
        pressure_scaled, imu_scaled = grad_outputs[0], grad_outputs[1]
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
            use_graph_pressure=_to_bool(checkpoint_model_config.get("use_graph_pressure", True), default=True),
            use_single_attention=_to_bool(checkpoint_model_config.get("use_single_attention", False), default=False),
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
    inference_start = time.perf_counter()

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

    inference_elapsed_s = max(time.perf_counter() - inference_start, 1e-12)
    predicted_frames = int(len(valid_starts))
    inference_fps = float(predicted_frames / inference_elapsed_s)

    final_predictions = torch.cat(all_predictions, dim=0).numpy()

    if "skeleton_scaler_mean" in checkpoint and "skeleton_scaler_scale" in checkpoint:
        skel_mean = np.asarray(checkpoint["skeleton_scaler_mean"], dtype=np.float32)
        skel_scale = np.asarray(checkpoint["skeleton_scaler_scale"], dtype=np.float32)
        final_predictions = final_predictions * skel_scale + skel_mean
        print("Applied skeleton inverse-transform (StandardScaler).")

    target_column_names = checkpoint.get("target_column_names", None)
    target_position_columns = checkpoint.get("target_position_columns", None)
    final_predictions = _apply_subject_height_denorm(
        predictions=final_predictions,
        frame_indices=output_frame_indices,
        segment_ids=segment_ids,
        data_pairs=skeleton_insole_datapath_pairs,
        target_column_names=target_column_names,
        target_position_columns=target_position_columns,
    )

    print(f"Prediction finished. Output shape: {final_predictions.shape}")

    split_outputs = _split_predictions_by_segment(
        predictions=final_predictions,
        frame_indices=output_frame_indices,
        segment_ids=segment_ids,
        data_pairs=skeleton_insole_datapath_pairs,
    )
    cycle_id = _next_prediction_cycle_id(Path(".") / "results" / "output")
    cycle_tag = f"cycle_{cycle_id:04d}"

    print(
        f"Prediction cycle {cycle_tag}: saving {len(split_outputs)} prediction file(s), "
        f"one per test tag."
    )
    saved_outputs = []
    for row in split_outputs:
        stem = join_nonempty(cycle_tag, abl_tag, row["tag"], model_mode)
        output_file = save_predictions(
            row["predictions"],
            args.model,
            frame_indices=row["frame_indices"],
            output_stem=stem,
            column_names=target_column_names,
        )
        saved_outputs.append(
            {
                "tag": row["tag"],
                "frame_indices": np.asarray(row["frame_indices"], dtype=np.int64),
                "output_file": output_file,
            }
        )

    return {
        "cycle_tag": cycle_tag,
        "model_mode": model_mode,
        "predicted_frames": predicted_frames,
        "inference_seconds": float(inference_elapsed_s),
        "inference_fps": inference_fps,
        "outputs": saved_outputs,
    }


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description="Prediction Processor")

    parser.add_argument("--model", choices=["transformer_encoder", "transformer", "BERT"], default="transformer_encoder", help="Model selection")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML file")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--abl_id", type=str, default=None)

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
    parser.add_argument("--soleformer_use_graph_pressure", type=str, default=None)
    parser.add_argument("--soleformer_use_single_attention", type=str, default=None)

    return parser
