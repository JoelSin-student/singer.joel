# Training processor
import argparse
import csv
import math
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from sources.loader import (
    PressureSkeletonDataset,
    PressureSkeletonSequenceDataset,
    calculate_grad,
    get_datapath_pairs,
    iter_ordered_data_pairs,
    load_and_combine_data,
    load_config,
    restructure_insole_data,
)
from sources.model import (
    AccelNet,
    DoubleCycleConsistencyLoss,
    PressNet,
    SoleFormer,
    Skeleton_Loss,
    Transformer_Encoder,
    Transformer_Encoder_Seq2Seq,
    train_mse,
    train_mse_with_cycle,
    pretrain_accelnet,
    pretrain_pressnet,
)
from sources.util import format_ablation_tag, join_nonempty, print_config, resolve_ablation_id


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


def _mode_value(config_train, model_mode, shared_key, mode_key):
    """Resolve mode-specific hyperparameter with fallback to shared key."""
    if model_mode == "soleformer" and mode_key in config_train and config_train.get(mode_key) is not None:
        return config_train[mode_key]
    return config_train[shared_key]


def _extract_subject_key_from_tag(tag):
    tag_str = str(tag).strip()
    if not tag_str:
        return ""
    return tag_str.split("_", 1)[0]


def _extract_situation_key_from_tag(tag):
    tag_str = str(tag).strip()
    if not tag_str:
        return ""

    parts = tag_str.split("_")
    if len(parts) < 2:
        return ""

    if parts[-1].lower() in {"training", "test"}:
        parts = parts[:-1]
    return "_".join(parts[1:])


def _build_segment_group_maps(data_pairs, order_by_situation=True):
    sorted_tags = [tag for tag, _ in iter_ordered_data_pairs(data_pairs, order_by_situation=order_by_situation)]
    segment_subject = {}
    segment_situation = {}
    for segment_id, tag in enumerate(sorted_tags, start=1):
        segment_subject[segment_id] = _extract_subject_key_from_tag(tag)
        segment_situation[segment_id] = _extract_situation_key_from_tag(tag)
    return segment_subject, segment_situation


def _build_group_kfold_segment_splits(
    segment_ids,
    segment_subject_map,
    segment_situation_map,
    n_splits,
    group_by="subject",
):
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if segment_ids.ndim != 1 or len(segment_ids) == 0:
        raise ValueError("segment_ids must be a non-empty 1D array for Group K-fold splitting")

    unique_segments = np.asarray(sorted(set(segment_ids.tolist())), dtype=np.int64)
    group_by = str(group_by).strip().lower()
    if group_by not in {"subject", "situation", "segment"}:
        raise ValueError("cv_group_by must be one of: 'subject', 'situation', 'segment'")

    group_labels = []
    for seg in unique_segments:
        if group_by == "subject":
            subject_key = segment_subject_map.get(int(seg), "")
            if not subject_key:
                raise KeyError(f"Missing subject mapping for segment id {int(seg)}")
            group_labels.append(str(subject_key))
        elif group_by == "situation":
            situation_key = segment_situation_map.get(int(seg), "")
            if not situation_key:
                raise KeyError(f"Missing situation mapping for segment id {int(seg)}")
            group_labels.append(str(situation_key))
        else:
            group_labels.append(f"segment_{int(seg)}")

    unique_groups = sorted(set(group_labels))
    if len(unique_groups) < 2:
        raise ValueError(
            f"Need at least 2 unique groups for Group K-fold, got {len(unique_groups)} with cv_group_by={group_by}."
        )

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("cv_n_splits must be at least 2")
    if n_splits > len(unique_groups):
        raise ValueError(
            f"cv_n_splits={n_splits} exceeds available unique groups={len(unique_groups)} for cv_group_by={group_by}."
        )

    group_labels = np.asarray(group_labels, dtype=object)
    splitter = GroupKFold(n_splits=n_splits)
    splits = []
    for fold_idx, (train_seg_idx, val_seg_idx) in enumerate(
        splitter.split(unique_segments, groups=group_labels), start=1
    ):
        train_segments = unique_segments[train_seg_idx]
        val_segments = unique_segments[val_seg_idx]
        train_mask = np.isin(segment_ids, train_segments)
        val_mask = np.isin(segment_ids, val_segments)

        if not train_mask.any() or not val_mask.any():
            raise RuntimeError(f"Invalid fold {fold_idx}: empty train or validation partition")

        train_subjects = sorted(
            {
                str(segment_subject_map.get(int(seg), ""))
                for seg in train_segments
                if segment_subject_map.get(int(seg), "")
            }
        )
        val_subjects = sorted(
            {
                str(segment_subject_map.get(int(seg), ""))
                for seg in val_segments
                if segment_subject_map.get(int(seg), "")
            }
        )
        train_situations = sorted(
            {
                str(segment_situation_map.get(int(seg), ""))
                for seg in train_segments
                if segment_situation_map.get(int(seg), "")
            }
        )
        val_situations = sorted(
            {
                str(segment_situation_map.get(int(seg), ""))
                for seg in val_segments
                if segment_situation_map.get(int(seg), "")
            }
        )

        splits.append(
            {
                "fold_idx": fold_idx,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "train_segments": train_segments,
                "val_segments": val_segments,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "train_situations": train_situations,
                "val_situations": val_situations,
                "train_groups": train_situations if group_by == "situation" else train_subjects if group_by == "subject" else [f"segment_{int(seg)}" for seg in train_segments],
                "val_groups": val_situations if group_by == "situation" else val_subjects if group_by == "subject" else [f"segment_{int(seg)}" for seg in val_segments],
            }
        )

    return splits


class WarmupCosineWeightDecayScheduler:
    """Epoch-based warmup + cosine decay scheduler for optimizer weight decay."""

    def __init__(self, optimizer, base_weight_decay, min_weight_decay, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_weight_decay = float(base_weight_decay)
        self.min_weight_decay = float(min_weight_decay)
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.total_epochs = int(max(1, total_epochs))

    def _compute(self, epoch_idx):
        epoch_idx = int(max(0, epoch_idx))

        if self.warmup_epochs > 0 and epoch_idx < self.warmup_epochs:
            # Linear warm-up from 0 to base weight decay.
            return self.base_weight_decay * float(epoch_idx + 1) / float(self.warmup_epochs)

        if self.total_epochs <= self.warmup_epochs + 1:
            return self.base_weight_decay

        progress = float(epoch_idx - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_weight_decay + (self.base_weight_decay - self.min_weight_decay) * cosine

    def step(self, epoch_idx):
        current_wd = float(self._compute(epoch_idx))
        for group in self.optimizer.param_groups:
            if group.get("apply_wd_schedule", False):
                group["weight_decay"] = current_wd
        return current_wd


def _build_linear_weight_decay_param_groups(modules, weight_decay):
    """Apply weight decay only to nn.Linear weights; all other params are no-decay."""
    decay_params = []
    no_decay_params = []
    decay_param_ids = set()

    for module in modules:
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Linear) and submodule.weight is not None and submodule.weight.requires_grad:
                param = submodule.weight
                param_id = id(param)
                if param_id not in decay_param_ids:
                    decay_params.append(param)
                    decay_param_ids.add(param_id)

    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            if id(param) in decay_param_ids:
                continue
            no_decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "weight_decay": float(weight_decay),
                "apply_wd_schedule": True,
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "apply_wd_schedule": False,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters were found for optimizer setup.")

    return param_groups


def start(args):
    config = load_config(args, args.config, args.model)

    model_mode = str(config["train"].get("model_mode", "simple_seq2seq")).lower()
    if model_mode not in {"original", "simple_seq2seq", "soleformer"}:
        raise ValueError("train.model_mode must be one of: original, simple_seq2seq, soleformer")

    abl_id = resolve_ablation_id(config, "train")
    abl_tag = format_ablation_tag(abl_id)

    train_cfg = config["train"]
    curriculum_by_situation = _to_bool(train_cfg.get("curriculum_by_situation", True), default=True)
    use_time_feature = _to_bool(train_cfg.get("use_time_feature", False), default=False)
    use_gradient_data = _to_bool(train_cfg.get("use_gradient_data", False), default=False)
    if model_mode == "soleformer" and use_gradient_data:
        print("SoleFormer mode: forcing use_gradient_data=False (derivative feature expansion is disabled).")
        use_gradient_data = False
    cv_enable = _to_bool(train_cfg.get("cv_enable", False), default=False)
    cv_n_splits = int(train_cfg.get("cv_n_splits", 5))
    cv_group_by = str(train_cfg.get("cv_group_by", "subject")).strip().lower()
    cv_refit_full_model = _to_bool(train_cfg.get("cv_refit_full_model", True), default=True)
    validation_ratio = float(train_cfg.get("validation_ratio", 0.2))
    shuffle_train_loader = _to_bool(train_cfg.get("shuffle_train_loader", not curriculum_by_situation), default=(not curriculum_by_situation))
    if curriculum_by_situation and shuffle_train_loader:
        print("curriculum_by_situation=True forces shuffle_train_loader=False to preserve situation order.")
        shuffle_train_loader = False

    skeleton_dir = os.path.join(config["location"]["data_path"], "skeleton")
    insole_dir = os.path.join(config["location"]["data_path"], "Insole")

    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)
    skeleton_df, insole_df, segment_ids = load_and_combine_data(
        skeleton_insole_datapath_pairs,
        order_by_situation=curriculum_by_situation,
    )
    target_df = skeleton_df

    pressure_lr_df, imu_lr_df, time_feature_df = restructure_insole_data(insole_df)
    target_df = target_df.bfill().ffill()

    sigma = float(train_cfg.get("smoothing_sigma", 0.0))
    if sigma > 0:
        pressure_lr_df = pressure_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        imu_lr_df = imu_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        print(f"Applied Gaussian smoothing before scaling (sigma={sigma}).")
    else:
        print("Gaussian smoothing disabled before scaling (smoothing_sigma=0).")

    # Resolve cycle loss and pretraining flags with model_mode awareness.
    if model_mode == "soleformer":
        use_cycle_loss = _to_bool(
            train_cfg.get("soleformer_use_cycle_loss", train_cfg.get("use_cycle_loss", True)),
            default=True,
        )
        pretrain_accelnet_enabled = _to_bool(
            train_cfg.get("soleformer_pretrain_accelnet", train_cfg.get("pretrain_accelnet", True)),
            default=True,
        )
        pretrain_pressnet_enabled = _to_bool(
            train_cfg.get("soleformer_pretrain_pressnet", train_cfg.get("pretrain_pressnet", True)),
            default=True,
        )
    else:
        use_cycle_loss = False
        pretrain_accelnet_enabled = False
        pretrain_pressnet_enabled = False

    enable_imu_cycle_loss = _to_bool(train_cfg.get("enable_imu_cycle_loss", True), default=True)
    enable_pressure_cycle_loss = _to_bool(train_cfg.get("enable_pressure_cycle_loss", True), default=True)
    freeze_pretrained_cycle_nets = _to_bool(train_cfg.get("freeze_pretrained_cycle_nets", True), default=True)
    accelnet_pretrained_path = train_cfg.get("accelnet_pretrained_path", None)
    pressnet_pretrained_path = train_cfg.get("pressnet_pretrained_path", None)
    pretrain_epochs = int(train_cfg.get("pretrain_epochs", 30))
    pretrain_learning_rate = float(train_cfg.get("pretrain_learning_rate", 0.001))
    pose_loss_mode = str(train_cfg.get("pose_loss_mode", "both")).strip().lower()
    pose_loss_weight_2d = float(train_cfg.get("pose_loss_weight_2d", 1.0))
    pose_loss_weight_3d = float(train_cfg.get("pose_loss_weight_3d", 1.0))
    imu_cycle_loss_weight = float(train_cfg.get("imu_cycle_loss_weight", 0.5))
    pressure_cycle_loss_weight = float(train_cfg.get("pressure_cycle_loss_weight", 0.5))
    if model_mode == "soleformer":
        use_lower_leg_angles_for_accelnet = _to_bool(
            train_cfg.get("soleformer_use_lower_leg_angles_for_accelnet", False),
            default=False,
        )
        use_graph_pressure = _to_bool(
            train_cfg.get("soleformer_use_graph_pressure", train_cfg.get("use_graph_pressure", True)),
            default=True,
        )
        use_single_attention = _to_bool(
            train_cfg.get("soleformer_use_single_attention", train_cfg.get("use_single_attention", False)),
            default=False,
        )
        use_weight_decay_schedule = _to_bool(
            train_cfg.get("soleformer_use_weight_decay_schedule", False),
            default=False,
        )
    else:
        use_lower_leg_angles_for_accelnet = _to_bool(
            train_cfg.get("use_lower_leg_angles_for_accelnet", False),
            default=False,
        )
        use_graph_pressure = True
        use_single_attention = False
        use_weight_decay_schedule = _to_bool(
            train_cfg.get("use_weight_decay_schedule", False),
            default=False,
        )

    weight_decay_warmup_epochs = int(
        train_cfg.get("soleformer_weight_decay_warmup_epochs", 5)
        if model_mode == "soleformer"
        else train_cfg.get("weight_decay_warmup_epochs", 0)
    )
    min_weight_decay = float(
        train_cfg.get("soleformer_min_weight_decay", 0.0)
        if model_mode == "soleformer"
        else train_cfg.get("min_weight_decay", 0.0)
    )

    grad_window_length = int(train_cfg.get("grad_window_length", 5))
    grad_polyorder = int(train_cfg.get("grad_polyorder", 2))
    grad_smooth_grad1 = _to_bool(train_cfg.get("grad_smooth_grad1", False), default=False)

    segment_subject_map, segment_situation_map = _build_segment_group_maps(
        skeleton_insole_datapath_pairs,
        order_by_situation=curriculum_by_situation,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def _run_training_split(
        split_label,
        train_pressure,
        val_pressure,
        train_imu,
        val_imu,
        train_target,
        val_target,
        train_time,
        val_time,
        train_segments,
        val_segments,
        split_meta=None,
    ):
        split_meta = split_meta or {}
        split_suffix = str(split_label).strip()
        split_display = split_suffix if split_suffix else "holdout"
        print(f"\n=== Training split: {split_display} ===")

        train_segments = np.asarray(train_segments, dtype=np.int64)
        val_segments = np.asarray(val_segments, dtype=np.int64)

        pressure_scaler = MinMaxScaler()
        imu_scaler = MinMaxScaler()
        skeleton_scaler = StandardScaler()

        train_pressure_np = train_pressure.to_numpy()
        val_pressure_np = val_pressure.to_numpy()
        train_imu_np = train_imu.to_numpy()
        val_imu_np = val_imu.to_numpy()

        train_pressure_scaled = pressure_scaler.fit_transform(train_pressure_np)
        val_pressure_scaled = pressure_scaler.transform(val_pressure_np)
        train_imu_scaled = imu_scaler.fit_transform(train_imu_np)
        val_imu_scaled = imu_scaler.transform(val_imu_np)

        if use_time_feature:
            from sources.loader import normalize_time_feature_per_segment

            train_time_scaled = normalize_time_feature_per_segment(train_time, train_segments)
            val_time_scaled = normalize_time_feature_per_segment(val_time, val_segments)
        else:
            train_time_scaled = None
            val_time_scaled = None

        grad_feature_stats = None
        if use_gradient_data:
            train_pressure_scaled, train_imu_scaled, grad_feature_stats = calculate_grad(
                train_pressure_scaled,
                train_imu_scaled,
                window_length=grad_window_length,
                polyorder=grad_polyorder,
                smooth_grad1=grad_smooth_grad1,
                return_stats=True,
            )
            val_pressure_scaled, val_imu_scaled = calculate_grad(
                val_pressure_scaled,
                val_imu_scaled,
                window_length=grad_window_length,
                polyorder=grad_polyorder,
                smooth_grad1=grad_smooth_grad1,
                normalization_stats=grad_feature_stats,
            )
            print(
                f"Derivative features enabled: input dim expanded from "
                f"{pressure_lr_df.shape[1] + imu_lr_df.shape[1]} to "
                f"{train_pressure_scaled.shape[1] + train_imu_scaled.shape[1]}."
            )

        train_skeleton_scaled = skeleton_scaler.fit_transform(train_target.to_numpy())
        val_skeleton_scaled = skeleton_scaler.transform(val_target.to_numpy())

        train_feature_parts = [train_pressure_scaled, train_imu_scaled]
        val_feature_parts = [val_pressure_scaled, val_imu_scaled]
        if use_time_feature:
            train_feature_parts.append(train_time_scaled)
            val_feature_parts.append(val_time_scaled)
            print(
                f"Time feature enabled: input dim expanded from "
                f"{train_pressure_scaled.shape[1] + train_imu_scaled.shape[1]} to "
                f"{train_pressure_scaled.shape[1] + train_imu_scaled.shape[1] + train_time_scaled.shape[1]}."
            )

        train_input_feature = np.concatenate(train_feature_parts, axis=1)
        val_input_feature = np.concatenate(val_feature_parts, axis=1)

        parameters = {
            "model_mode": model_mode,
            "curriculum_by_situation": curriculum_by_situation,
            "shuffle_train_loader": shuffle_train_loader,
            "use_gradient_data": use_gradient_data,
            "use_time_feature": use_time_feature,
            "use_cycle_loss": use_cycle_loss,
            "enable_imu_cycle_loss": enable_imu_cycle_loss,
            "enable_pressure_cycle_loss": enable_pressure_cycle_loss,
            "freeze_pretrained_cycle_nets": freeze_pretrained_cycle_nets,
            "pose_loss_weight_2d": pose_loss_weight_2d,
            "pose_loss_weight_3d": pose_loss_weight_3d,
            "pose_loss_mode": pose_loss_mode,
            "imu_cycle_loss_weight": imu_cycle_loss_weight,
            "pressure_cycle_loss_weight": pressure_cycle_loss_weight,
            "use_lower_leg_angles_for_accelnet": bool(use_lower_leg_angles_for_accelnet and model_mode == "soleformer"),
            "use_graph_pressure": bool(use_graph_pressure and model_mode == "soleformer"),
            "use_single_attention": bool(use_single_attention and model_mode == "soleformer"),
            "use_weight_decay_schedule": bool(use_weight_decay_schedule and model_mode == "soleformer"),
            "weight_decay_warmup_epochs": weight_decay_warmup_epochs,
            "min_weight_decay": min_weight_decay,
            "d_model": int(_mode_value(train_cfg, model_mode, "d_model", "soleformer_d_model")),
            "n_head": int(_mode_value(train_cfg, model_mode, "n_head", "soleformer_n_head")),
            "num_encoder_layer": int(_mode_value(train_cfg, model_mode, "num_encoder_layer", "soleformer_num_encoder_layer")),
            "dropout": float(_mode_value(train_cfg, model_mode, "dropout", "soleformer_dropout")),
            "num_epoch": int(_mode_value(train_cfg, model_mode, "epoch", "soleformer_epoch")),
            "batch_size": int(_mode_value(train_cfg, model_mode, "batch_size", "soleformer_batch_size")),
            "learning_rate": float(_mode_value(train_cfg, model_mode, "learning_rate", "soleformer_learning_rate")),
            "weight_decay": float(_mode_value(train_cfg, model_mode, "weight_decay", "soleformer_weight_decay")),
            "sequence_len": int(_mode_value(train_cfg, model_mode, "sequence_len", "soleformer_sequence_len")),
            "input_dim": train_input_feature.shape[1],
            "output_dim": target_df.shape[1],
            "num_joints": target_df.shape[1] // 3,
            "num_dims": 3 if target_df.shape[1] % 3 == 0 else 1,
        }

        print_config(parameters)

        if parameters["model_mode"] in {"simple_seq2seq", "soleformer"}:
            train_dataset = PressureSkeletonSequenceDataset(
                train_input_feature,
                train_skeleton_scaled,
                sequence_length=parameters["sequence_len"],
                segment_ids=train_segments,
            )
            val_dataset = PressureSkeletonSequenceDataset(
                val_input_feature,
                val_skeleton_scaled,
                sequence_length=parameters["sequence_len"],
                segment_ids=val_segments,
            )
        else:
            train_dataset = PressureSkeletonDataset(
                train_input_feature,
                train_skeleton_scaled,
                sequence_length=parameters["sequence_len"],
                segment_ids=train_segments,
            )
            val_dataset = PressureSkeletonDataset(
                val_input_feature,
                val_skeleton_scaled,
                sequence_length=parameters["sequence_len"],
                segment_ids=val_segments,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=parameters["batch_size"],
            shuffle=parameters["shuffle_train_loader"],
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=parameters["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        if parameters["model_mode"] == "simple_seq2seq":
            model = Transformer_Encoder_Seq2Seq(
                input_dim=parameters["input_dim"],
                d_model=parameters["d_model"],
                nhead=parameters["n_head"],
                num_encoder_layers=parameters["num_encoder_layer"],
                num_joints=parameters["num_joints"],
                num_dims=parameters["num_dims"],
                dropout=parameters["dropout"],
            ).to(device)
        elif parameters["model_mode"] == "soleformer":
            model = SoleFormer(
                pressure_dim=train_pressure_scaled.shape[1],
                imu_dim=train_imu_scaled.shape[1],
                d_model=parameters["d_model"],
                nhead=parameters["n_head"],
                num_encoder_layers=parameters["num_encoder_layer"],
                output_dim=parameters["output_dim"],
                dropout=parameters["dropout"],
                use_graph_pressure=parameters["use_graph_pressure"],
                use_single_attention=parameters["use_single_attention"],
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

        cycle_training_active = bool(parameters["model_mode"] == "soleformer" and parameters["use_cycle_loss"])
        accel_net = None
        press_net = None
        accelnet_loaded = False
        pressnet_loaded = False

        local_accelnet_pretrained_path = accelnet_pretrained_path
        local_pressnet_pretrained_path = pressnet_pretrained_path

        if cycle_training_active:
            accel_input_dim = parameters["output_dim"]
            if parameters["use_lower_leg_angles_for_accelnet"]:
                accel_input_dim = 6

            accel_net = AccelNet(
                input_dim=accel_input_dim,
                output_dim=int(train_imu_scaled.shape[1]),
                dropout=parameters["dropout"],
            ).to(device)
            accel_net._pressure_dim = int(train_pressure_scaled.shape[1])
            accel_net._imu_dim = int(train_imu_scaled.shape[1])
            accel_net._imu_start = int(train_pressure_scaled.shape[1])
            accel_net._use_lower_leg_angles_for_accelnet = bool(parameters["use_lower_leg_angles_for_accelnet"])
            accel_net._foot_orientation_indices = (17, 18, 21, 22) if parameters["num_joints"] >= 23 else None
            press_net = PressNet(
                input_dim=parameters["output_dim"],
                output_dim=int(train_pressure_scaled.shape[1]),
                dropout=parameters["dropout"],
            ).to(device)
            press_net._pressure_dim = int(train_pressure_scaled.shape[1])

            if pretrain_accelnet_enabled and not local_accelnet_pretrained_path:
                accelnet_save_path = os.path.join(".", "results", "pretrained_aux", "accelnet_pretrained.pt")
                os.makedirs(os.path.dirname(accelnet_save_path), exist_ok=True)
                print("\n" + "=" * 60)
                print("PRETRAINING AccelNet (pose -> 6DoF IMU)...")
                print("=" * 60)
                pretrain_accelnet(
                    accel_net,
                    train_loader,
                    val_loader,
                    num_epochs=pretrain_epochs,
                    learning_rate=pretrain_learning_rate,
                    save_path=accelnet_save_path,
                    device=device,
                )
                local_accelnet_pretrained_path = accelnet_save_path
                print(f"Using newly pretrained AccelNet from {accelnet_save_path}")

            if pretrain_pressnet_enabled and not local_pressnet_pretrained_path:
                pressnet_save_path = os.path.join(".", "results", "pretrained_aux", "pressnet_pretrained.pt")
                os.makedirs(os.path.dirname(pressnet_save_path), exist_ok=True)
                print("\n" + "=" * 60)
                print("PRETRAINING PressNet (pose -> foot pressure)...")
                print("=" * 60)
                pretrain_pressnet(
                    press_net,
                    train_loader,
                    val_loader,
                    num_epochs=pretrain_epochs,
                    learning_rate=pretrain_learning_rate,
                    save_path=pressnet_save_path,
                    device=device,
                )
                local_pressnet_pretrained_path = pressnet_save_path
                print(f"Using newly pretrained PressNet from {pressnet_save_path}")

            if local_accelnet_pretrained_path:
                if not os.path.isfile(local_accelnet_pretrained_path):
                    raise FileNotFoundError(f"AccelNet checkpoint not found: {local_accelnet_pretrained_path}")
                accel_ckpt = torch.load(local_accelnet_pretrained_path, map_location="cpu")
                accel_state = accel_ckpt["model_state_dict"] if isinstance(accel_ckpt, dict) and "model_state_dict" in accel_ckpt else accel_ckpt
                accel_net.load_state_dict(accel_state, strict=True)
                accelnet_loaded = True

            if local_pressnet_pretrained_path:
                if not os.path.isfile(local_pressnet_pretrained_path):
                    raise FileNotFoundError(f"PressNet checkpoint not found: {local_pressnet_pretrained_path}")
                press_ckpt = torch.load(local_pressnet_pretrained_path, map_location="cpu")
                press_state = press_ckpt["model_state_dict"] if isinstance(press_ckpt, dict) and "model_state_dict" in press_ckpt else press_ckpt
                press_net.load_state_dict(press_state, strict=True)
                pressnet_loaded = True

            if freeze_pretrained_cycle_nets:
                for p in accel_net.parameters():
                    p.requires_grad = False
                for p in press_net.parameters():
                    p.requires_grad = False
                accel_net.eval()
                press_net.eval()

            criterion = DoubleCycleConsistencyLoss(
                accel_net=accel_net,
                press_net=press_net,
                weight_pose=1.0,
                weight_imu_cycle=parameters["imu_cycle_loss_weight"],
                weight_pressure_cycle=parameters["pressure_cycle_loss_weight"],
                weight_2d_loss=parameters["pose_loss_weight_2d"],
                weight_3d_loss=parameters["pose_loss_weight_3d"],
                pose_loss_mode=parameters["pose_loss_mode"],
                enable_imu_cycle=parameters["enable_imu_cycle_loss"],
                enable_pressure_cycle=parameters["enable_pressure_cycle_loss"],
                use_lower_leg_angles_for_accelnet=parameters["use_lower_leg_angles_for_accelnet"],
                accelnet_foot_indices=getattr(accel_net, "_foot_orientation_indices", None),
            )

            print(
                f"Cycle training enabled. "
                f"AccelNet loaded={accelnet_loaded}, PressNet loaded={pressnet_loaded}, "
                f"cycle_nets_frozen={freeze_pretrained_cycle_nets}."
            )
        else:
            criterion = Skeleton_Loss()

        trainable_modules = [model]
        if cycle_training_active and not freeze_pretrained_cycle_nets:
            trainable_modules.extend([accel_net, press_net])

        param_groups = _build_linear_weight_decay_param_groups(
            modules=trainable_modules,
            weight_decay=parameters["weight_decay"],
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=parameters["learning_rate"],
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        wd_scheduler = None
        if parameters["use_weight_decay_schedule"]:
            wd_scheduler = WarmupCosineWeightDecayScheduler(
                optimizer=optimizer,
                base_weight_decay=parameters["weight_decay"],
                min_weight_decay=parameters["min_weight_decay"],
                warmup_epochs=parameters["weight_decay_warmup_epochs"],
                total_epochs=parameters["num_epoch"],
            )

        checkpoint_extra = {
            "skeleton_scaler_mean": skeleton_scaler.mean_.tolist(),
            "skeleton_scaler_scale": skeleton_scaler.scale_.tolist(),
            "pressure_scaler_min": pressure_scaler.min_.tolist(),
            "pressure_scaler_scale": pressure_scaler.scale_.tolist(),
            "pressure_scaler_data_min": pressure_scaler.data_min_.tolist(),
            "pressure_scaler_data_max": pressure_scaler.data_max_.tolist(),
            "pressure_scaler_data_range": pressure_scaler.data_range_.tolist(),
            "pressure_scaler_n_features": int(pressure_scaler.n_features_in_),
            "imu_scaler_min": imu_scaler.min_.tolist(),
            "imu_scaler_scale": imu_scaler.scale_.tolist(),
            "imu_scaler_data_min": imu_scaler.data_min_.tolist(),
            "imu_scaler_data_max": imu_scaler.data_max_.tolist(),
            "imu_scaler_data_range": imu_scaler.data_range_.tolist(),
            "imu_scaler_n_features": int(imu_scaler.n_features_in_),
            "preprocessing_grad_window_length": grad_window_length,
            "preprocessing_grad_polyorder": grad_polyorder,
            "preprocessing_grad_smooth_grad1": grad_smooth_grad1,
            "model_mode": parameters["model_mode"],
            "abl_id": abl_id,
            "split_label": split_display,
            "target_column_names": list(target_df.columns),
            "target_output_dim": int(target_df.shape[1]),
            "train_use_cycle_loss": bool(cycle_training_active),
            "train_enable_imu_cycle_loss": bool(enable_imu_cycle_loss),
            "train_enable_pressure_cycle_loss": bool(enable_pressure_cycle_loss),
            "train_pose_loss_weight_2d": pose_loss_weight_2d,
            "train_pose_loss_weight_3d": pose_loss_weight_3d,
            "train_imu_cycle_loss_weight": imu_cycle_loss_weight,
            "train_pressure_cycle_loss_weight": pressure_cycle_loss_weight,
            "train_use_lower_leg_angles_for_accelnet": bool(parameters["use_lower_leg_angles_for_accelnet"]),
            "train_use_graph_pressure": bool(parameters["use_graph_pressure"]),
            "train_use_single_attention": bool(parameters["use_single_attention"]),
            "train_use_weight_decay_schedule": bool(parameters["use_weight_decay_schedule"]),
            "train_weight_decay_warmup_epochs": int(parameters["weight_decay_warmup_epochs"]),
            "train_min_weight_decay": float(parameters["min_weight_decay"]),
            "train_freeze_pretrained_cycle_nets": bool(freeze_pretrained_cycle_nets),
            "accelnet_pretrained_path": local_accelnet_pretrained_path,
            "pressnet_pretrained_path": local_pressnet_pretrained_path,
            "accelnet_pretrained_loaded": bool(accelnet_loaded),
            "pressnet_pretrained_loaded": bool(pressnet_loaded),
            "cv_enable": bool(cv_enable),
            "cv_group_by": cv_group_by,
            "cv_n_splits": int(cv_n_splits),
            "curriculum_by_situation": bool(curriculum_by_situation),
        }
        checkpoint_extra.update(split_meta)

        checkpoint_extra["preprocessing_use_time_feature"] = bool(use_time_feature)
        checkpoint_extra["preprocessing_use_gradient_data"] = bool(use_gradient_data)
        if grad_feature_stats is not None:
            checkpoint_extra.update(
                {
                    "grad_pressure_mean": grad_feature_stats["pressure_mean"].tolist(),
                    "grad_pressure_std": grad_feature_stats["pressure_std"].tolist(),
                    "grad_imu_mean": grad_feature_stats["imu_mean"].tolist(),
                    "grad_imu_std": grad_feature_stats["imu_std"].tolist(),
                }
            )

        best_ckpt_name = join_nonempty("best_skeleton_model", abl_tag, parameters["model_mode"], split_suffix)
        final_ckpt_name = join_nonempty("final_skeleton_model", abl_tag, parameters["model_mode"], split_suffix)
        best_ckpt_path = os.path.join(".", "results", "weight", f"{best_ckpt_name}.pth")
        final_ckpt_path = os.path.join(".", "results", "weight", f"{final_ckpt_name}.pth")

        if cycle_training_active:
            loss_history = train_mse_with_cycle(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=parameters["num_epoch"],
                save_path=best_ckpt_path,
                device=device,
                checkpoint_extra=checkpoint_extra,
                wd_scheduler=wd_scheduler,
            )
        else:
            loss_history = train_mse(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=parameters["num_epoch"],
                save_path=best_ckpt_path,
                device=device,
                checkpoint_extra=checkpoint_extra,
                wd_scheduler=wd_scheduler,
            )

        learning_results_dir = os.path.join(".", "results", "learning_results")
        os.makedirs(learning_results_dir, exist_ok=True)
        learning_results_path = os.path.join(
            learning_results_dir,
            f"{join_nonempty('Learning_results', abl_tag, parameters['model_mode'], split_suffix)}.csv",
        )

        if loss_history:
            fieldnames = ["epoch", "train_loss", "val_loss"]
            component_keys = set()
            for row in loss_history:
                for key in row.keys():
                    if key not in fieldnames:
                        component_keys.add(key)
            fieldnames.extend(sorted(component_keys))

            with open(learning_results_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(loss_history)

            print(f"Saved learning curves to {learning_results_path}")
        else:
            print("Warning: loss history is empty; no learning results CSV was written.")

        final_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": {
                "input_dim": parameters["input_dim"],
                "d_model": parameters["d_model"],
                "nhead": parameters["n_head"],
                "num_encoder_layers": parameters["num_encoder_layer"],
                "num_joints": parameters["num_joints"],
                "output_dim": parameters["output_dim"],
                "pressure_dim": int(train_pressure_scaled.shape[1]),
                "imu_dim": int(train_imu_scaled.shape[1]),
                "model_mode": parameters["model_mode"],
                "abl_id": abl_id,
                "use_graph_pressure": bool(parameters["use_graph_pressure"]),
                "use_single_attention": bool(parameters["use_single_attention"]),
            },
            **checkpoint_extra,
        }
        if cycle_training_active:
            final_checkpoint["accel_net_state_dict"] = accel_net.state_dict()
            final_checkpoint["press_net_state_dict"] = press_net.state_dict()
        torch.save(final_checkpoint, final_ckpt_path)

        train_subjects = sorted(
            {
                str(segment_subject_map.get(int(seg), ""))
                for seg in train_segments
                if segment_subject_map.get(int(seg), "")
            }
        )
        val_subjects = sorted(
            {
                str(segment_subject_map.get(int(seg), ""))
                for seg in val_segments
                if segment_subject_map.get(int(seg), "")
            }
        )

        best_epoch = None
        best_val_loss = None
        if loss_history:
            valid_rows = [row for row in loss_history if "val_loss" in row]
            if valid_rows:
                best_row = min(valid_rows, key=lambda r: float(r.get("val_loss", np.inf)))
                best_epoch = int(best_row.get("epoch", -1))
                best_val_loss = float(best_row.get("val_loss", np.nan))

        return {
            "split_label": split_display,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "learning_results_path": learning_results_path,
            "best_ckpt_path": best_ckpt_path,
            "final_ckpt_path": final_ckpt_path,
            "n_train_rows": int(len(train_pressure)),
            "n_val_rows": int(len(val_pressure)),
            "n_train_subjects": int(len(train_subjects)),
            "n_val_subjects": int(len(val_subjects)),
            "train_subjects": "|".join(train_subjects),
            "val_subjects": "|".join(val_subjects),
        }

    if cv_enable:
        print(
            f"Group K-fold enabled: n_splits={cv_n_splits}, cv_group_by={cv_group_by}, "
            f"cv_refit_full_model={cv_refit_full_model}"
        )
        fold_defs = _build_group_kfold_segment_splits(
            segment_ids,
            segment_subject_map,
            segment_situation_map,
            n_splits=cv_n_splits,
            group_by=cv_group_by,
        )

        cv_rows = []
        for fold in fold_defs:
            train_mask = np.asarray(fold["train_mask"], dtype=bool)
            val_mask = np.asarray(fold["val_mask"], dtype=bool)
            fold_idx = int(fold["fold_idx"])
            split_label = f"fold_{fold_idx}"

            fold_result = _run_training_split(
                split_label=split_label,
                train_pressure=pressure_lr_df.loc[train_mask].reset_index(drop=True),
                val_pressure=pressure_lr_df.loc[val_mask].reset_index(drop=True),
                train_imu=imu_lr_df.loc[train_mask].reset_index(drop=True),
                val_imu=imu_lr_df.loc[val_mask].reset_index(drop=True),
                train_target=target_df.loc[train_mask].reset_index(drop=True),
                val_target=target_df.loc[val_mask].reset_index(drop=True),
                train_time=time_feature_df.loc[train_mask].reset_index(drop=True),
                val_time=time_feature_df.loc[val_mask].reset_index(drop=True),
                train_segments=np.asarray(segment_ids)[train_mask],
                val_segments=np.asarray(segment_ids)[val_mask],
                split_meta={
                    "cv_fold_index": fold_idx,
                    "cv_train_segments": int(len(fold["train_segments"])),
                    "cv_val_segments": int(len(fold["val_segments"])),
                    "cv_group_by": cv_group_by,
                    "cv_train_subjects": "|".join(fold["train_subjects"]),
                    "cv_val_subjects": "|".join(fold["val_subjects"]),
                    "cv_train_situations": "|".join(fold["train_situations"]),
                    "cv_val_situations": "|".join(fold["val_situations"]),
                    "cv_train_groups": "|".join(fold["train_groups"]),
                    "cv_val_groups": "|".join(fold["val_groups"]),
                },
            )
            fold_result["cv_fold_index"] = fold_idx
            fold_result["cv_train_segments"] = int(len(fold["train_segments"]))
            fold_result["cv_val_segments"] = int(len(fold["val_segments"]))
            fold_result["cv_group_by"] = cv_group_by
            cv_rows.append(fold_result)

        learning_results_dir = os.path.join(".", "results", "learning_results")
        os.makedirs(learning_results_dir, exist_ok=True)
        cv_summary_path = os.path.join(
            learning_results_dir,
            f"{join_nonempty('CV_summary', abl_tag, model_mode)}.csv",
        )
        if cv_rows:
            fieldnames = sorted(set().union(*(row.keys() for row in cv_rows)))
            with open(cv_summary_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(cv_rows)
            print(f"Saved CV summary to {cv_summary_path}")

        if cv_refit_full_model:
            print(
                "Starting optional full-data refit after CV. "
                "Validation stream equals training stream for optimization monitoring only."
            )
            _run_training_split(
                split_label="full_refit",
                train_pressure=pressure_lr_df.reset_index(drop=True),
                val_pressure=pressure_lr_df.reset_index(drop=True),
                train_imu=imu_lr_df.reset_index(drop=True),
                val_imu=imu_lr_df.reset_index(drop=True),
                train_target=target_df.reset_index(drop=True),
                val_target=target_df.reset_index(drop=True),
                train_time=time_feature_df.reset_index(drop=True),
                val_time=time_feature_df.reset_index(drop=True),
                train_segments=np.asarray(segment_ids).copy(),
                val_segments=np.asarray(segment_ids).copy(),
                split_meta={
                    "cv_refit_full_model": True,
                },
            )
    else:
        (
            train_pressure,
            val_pressure,
            train_imu,
            val_imu,
            train_target,
            val_target,
            train_time,
            val_time,
            train_segments,
            val_segments,
        ) = train_test_split(
            pressure_lr_df,
            imu_lr_df,
            target_df,
            time_feature_df,
            segment_ids,
            test_size=validation_ratio,
            shuffle=False,
        )

        _run_training_split(
            split_label="",
            train_pressure=train_pressure.reset_index(drop=True),
            val_pressure=val_pressure.reset_index(drop=True),
            train_imu=train_imu.reset_index(drop=True),
            val_imu=val_imu.reset_index(drop=True),
            train_target=train_target.reset_index(drop=True),
            val_target=val_target.reset_index(drop=True),
            train_time=train_time.reset_index(drop=True),
            val_time=val_time.reset_index(drop=True),
            train_segments=np.asarray(train_segments),
            val_segments=np.asarray(val_segments),
            split_meta={
                "validation_ratio": validation_ratio,
            },
        )


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description="Training Processor")

    parser.add_argument("--model", choices=["transformer_encoder", "transformer", "BERT"], default="transformer_encoder", help="Model selection")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML file")
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--num_encoder_layer", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sequence_len", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--smoothing_sigma", type=float, default=None)
    parser.add_argument("--validation_ratio", type=float, default=None)
    parser.add_argument("--model_mode", type=str, default=None, choices=["original", "simple_seq2seq", "soleformer"])
    parser.add_argument("--abl_id", type=str, default=None)
    parser.add_argument("--curriculum_by_situation", type=str, default=None)
    parser.add_argument("--shuffle_train_loader", type=str, default=None)
    parser.add_argument("--use_time_feature", type=str, default=None)
    parser.add_argument("--use_gradient_data", type=str, default=None)
    parser.add_argument("--use_cycle_loss", type=str, default=None)
    parser.add_argument("--enable_imu_cycle_loss", type=str, default=None)
    parser.add_argument("--enable_pressure_cycle_loss", type=str, default=None)
    parser.add_argument("--freeze_pretrained_cycle_nets", type=str, default=None)
    parser.add_argument("--pose_loss_weight_2d", type=float, default=None)
    parser.add_argument("--pose_loss_weight_3d", type=float, default=None)
    parser.add_argument("--pose_loss_mode", type=str, default=None, choices=["2d", "3d", "both", "none", "off", "disabled"])
    parser.add_argument("--imu_cycle_loss_weight", type=float, default=None)
    parser.add_argument("--pressure_cycle_loss_weight", type=float, default=None)
    parser.add_argument("--accelnet_pretrained_path", type=str, default=None)
    parser.add_argument("--pressnet_pretrained_path", type=str, default=None)
    parser.add_argument("--pretrain_accelnet", type=str, default=None, help="Enable AccelNet pretraining (true/false)")
    parser.add_argument("--pretrain_pressnet", type=str, default=None, help="Enable PressNet pretraining (true/false)")
    parser.add_argument("--pretrain_epochs", type=int, default=None, help="Epochs for auxiliary net pretraining")
    parser.add_argument("--pretrain_learning_rate", type=float, default=None, help="Learning rate for auxiliary net pretraining")
    parser.add_argument("--grad_window_length", type=int, default=None)
    parser.add_argument("--grad_polyorder", type=int, default=None)
    parser.add_argument("--grad_smooth_grad1", type=str, default=None)
    parser.add_argument("--cv_enable", type=str, default=None)
    parser.add_argument("--cv_n_splits", type=int, default=None)
    parser.add_argument("--cv_group_by", type=str, default=None, choices=["subject", "situation", "segment"])
    parser.add_argument("--cv_refit_full_model", type=str, default=None)

    # SoleFormer-only overrides (used when --model_mode soleformer)
    parser.add_argument("--soleformer_d_model", type=int, default=None)
    parser.add_argument("--soleformer_n_head", type=int, default=None)
    parser.add_argument("--soleformer_num_encoder_layer", type=int, default=None)
    parser.add_argument("--soleformer_dropout", type=float, default=None)
    parser.add_argument("--soleformer_epoch", type=int, default=None)
    parser.add_argument("--soleformer_batch_size", type=int, default=None)
    parser.add_argument("--soleformer_learning_rate", type=float, default=None)
    parser.add_argument("--soleformer_weight_decay", type=float, default=None)
    parser.add_argument("--soleformer_sequence_len", type=int, default=None)
    parser.add_argument("--soleformer_use_lower_leg_angles_for_accelnet", type=str, default=None)
    parser.add_argument("--soleformer_use_graph_pressure", type=str, default=None)
    parser.add_argument("--soleformer_use_single_attention", type=str, default=None)
    parser.add_argument("--soleformer_use_weight_decay_schedule", type=str, default=None)
    parser.add_argument("--soleformer_weight_decay_warmup_epochs", type=int, default=None)
    parser.add_argument("--soleformer_min_weight_decay", type=float, default=None)

    return parser
