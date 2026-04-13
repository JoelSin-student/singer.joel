# Training processor
import argparse
import csv
import math
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from notebooks.loader import (
    PressureSkeletonDataset,
    PressureSkeletonSequenceDataset,
    calculate_grad,
    get_datapath_pairs,
    load_awinda_targets_from_merged_csv,
    load_awinda_targets_from_converted_tabs,
    load_and_combine_data,
    load_config,
    restructure_insole_data,
)
from notebooks.model import (
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
from notebooks.util import format_ablation_tag, join_nonempty, print_config, resolve_ablation_id


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

    skeleton_dir = os.path.join(config["location"]["data_path"], "skeleton")
    insole_dir = os.path.join(config["location"]["data_path"], "Insole")

    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)
    skeleton_df, insole_df, segment_ids = load_and_combine_data(skeleton_insole_datapath_pairs)
    target_df = skeleton_df

    if model_mode == "soleformer":
        awinda_targets_dir = config["location"].get("awinda_targets_dir", None)
        awinda_tabs_dir = config["location"].get("awinda_tabs_dir", None)

        if not awinda_targets_dir and not awinda_tabs_dir:
            raise ValueError(
                "Set either location.awinda_targets_dir (merged AwindaTarget_*.csv) "
                "or location.awinda_tabs_dir (converted raw Awinda tabs) when "
                "train.model_mode=soleformer."
            )

        include_target_positions = _to_bool(config["train"].get("include_target_positions", True), default=True)
        include_target_joint_angles = _to_bool(config["train"].get("include_target_joint_angles", True), default=True)
        joint_angles_tab_suffix = str(config["train"].get("joint_angles_tab_suffix", "tab9_Joint_Angles_ZXY"))

        if awinda_targets_dir:
            print(f"Using merged soleformer targets from: {awinda_targets_dir}")
            target_df, awinda_target_meta = load_awinda_targets_from_merged_csv(
                skeleton_insole_datapath_pairs,
                awinda_targets_dir=awinda_targets_dir,
                include_positions=include_target_positions,
                include_joint_angles=include_target_joint_angles,
            )
        else:
            print(f"Using converted raw Awinda tabs from: {awinda_tabs_dir}")
            target_df, awinda_target_meta = load_awinda_targets_from_converted_tabs(
                skeleton_insole_datapath_pairs,
                awinda_tabs_dir=awinda_tabs_dir,
                include_positions=include_target_positions,
                include_joint_angles=include_target_joint_angles,
                joint_angles_suffix=joint_angles_tab_suffix,
            )
        target_df = target_df.bfill().ffill()

        if target_df.isna().any().any():
            nan_count = int(target_df.isna().sum().sum())
            raise ValueError(
                f"Awinda target preprocessing left {nan_count} NaN values after fill. "
                "Please inspect converted tabs for missing numeric values."
            )

        if not np.isfinite(target_df.to_numpy(dtype=np.float32)).all():
            raise ValueError(
                "Awinda targets contain non-finite values (inf/-inf). "
                "Please sanitize converted tabs before training soleformer mode."
            )

        if len(target_df) != len(segment_ids):
            raise ValueError(
                f"Awinda converted targets have {len(target_df)} rows but insole stream has {len(segment_ids)} rows. "
                "Please check tab conversion/synchronization before training soleformer mode."
            )
    else:
        awinda_target_meta = {
            "target_columns": list(skeleton_df.columns),
            "position_columns": list(skeleton_df.columns),
            "angle_columns": [],
            "joint_angles_suffix": "n/a",
        }

    pressure_lr_df, imu_lr_df, time_feature_df = restructure_insole_data(insole_df)

    target_df = target_df.bfill().ffill()

    sigma = float(config["train"].get("smoothing_sigma", 0.0))
    if sigma > 0:
        pressure_lr_df = pressure_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        imu_lr_df = imu_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        print(f"Applied Gaussian smoothing before scaling (sigma={sigma}).")
    else:
        print("Gaussian smoothing disabled before scaling (smoothing_sigma=0).")

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
        test_size=0.2,
        shuffle=False,
    )

    pressure_scaler = MinMaxScaler()
    imu_scaler = MinMaxScaler()
    skeleton_scaler = StandardScaler()

    train_pressure_scaled = pressure_scaler.fit_transform(train_pressure)
    val_pressure_scaled = pressure_scaler.transform(val_pressure)

    train_imu_scaled = imu_scaler.fit_transform(train_imu)
    val_imu_scaled = imu_scaler.transform(val_imu)

    use_time_feature = _to_bool(config["train"].get("use_time_feature", False), default=False)
    if use_time_feature:
        # Normalize time feature per segment to avoid encoding global recording offsets.
        from notebooks.loader import normalize_time_feature_per_segment
        train_time_scaled = normalize_time_feature_per_segment(train_time, train_segments)
        val_time_scaled = normalize_time_feature_per_segment(val_time, val_segments)
    else:
        train_time_scaled = None
        val_time_scaled = None

    use_gradient_data = _to_bool(config["train"].get("use_gradient_data", False), default=False)
    
    # Resolve cycle loss and pretraining flags with model_mode awareness
    if model_mode == "soleformer":
        use_cycle_loss = _to_bool(
            config["train"].get("soleformer_use_cycle_loss", config["train"].get("use_cycle_loss", True)),
            default=True,
        )
        pretrain_accelnet_enabled = _to_bool(
            config["train"].get("soleformer_pretrain_accelnet", config["train"].get("pretrain_accelnet", True)),
            default=True,
        )
        pretrain_pressnet_enabled = _to_bool(
            config["train"].get("soleformer_pretrain_pressnet", config["train"].get("pretrain_pressnet", True)),
            default=True,
        )
    else:
        use_cycle_loss = False  # Only soleformer supports cycle loss
        pretrain_accelnet_enabled = False
        pretrain_pressnet_enabled = False
    
    enable_imu_cycle_loss = _to_bool(config["train"].get("enable_imu_cycle_loss", True), default=True)
    enable_pressure_cycle_loss = _to_bool(config["train"].get("enable_pressure_cycle_loss", True), default=True)
    freeze_pretrained_cycle_nets = _to_bool(
        config["train"].get("freeze_pretrained_cycle_nets", True),
        default=True,
    )
    accelnet_pretrained_path = config["train"].get("accelnet_pretrained_path", None)
    pressnet_pretrained_path = config["train"].get("pressnet_pretrained_path", None)
    pretrain_epochs = int(config["train"].get("pretrain_epochs", 30))
    pretrain_learning_rate = float(config["train"].get("pretrain_learning_rate", 0.001))
    pose_loss_weight_2d = float(config["train"].get("pose_loss_weight_2d", 1.0))
    pose_loss_weight_3d = float(config["train"].get("pose_loss_weight_3d", 1.0))
    imu_cycle_loss_weight = float(config["train"].get("imu_cycle_loss_weight", 0.5))
    pressure_cycle_loss_weight = float(config["train"].get("pressure_cycle_loss_weight", 0.5))
    if model_mode == "soleformer":
        use_lower_leg_angles_for_accelnet = _to_bool(
            config["train"].get("soleformer_use_lower_leg_angles_for_accelnet", False),
            default=False,
        )
        use_weight_decay_schedule = _to_bool(
            config["train"].get("soleformer_use_weight_decay_schedule", False),
            default=False,
        )
    else:
        use_lower_leg_angles_for_accelnet = _to_bool(
            config["train"].get("use_lower_leg_angles_for_accelnet", False),
            default=False,
        )
        use_weight_decay_schedule = _to_bool(
            config["train"].get("use_weight_decay_schedule", False),
            default=False,
        )
    weight_decay_warmup_epochs = int(
        config["train"].get("soleformer_weight_decay_warmup_epochs", 5)
        if model_mode == "soleformer"
        else config["train"].get("weight_decay_warmup_epochs", 0)
    )
    min_weight_decay = float(
        config["train"].get("soleformer_min_weight_decay", 0.0)
        if model_mode == "soleformer"
        else config["train"].get("min_weight_decay", 0.0)
    )

    grad_window_length = int(config["train"].get("grad_window_length", 5))
    grad_polyorder = int(config["train"].get("grad_polyorder", 2))
    grad_smooth_grad1 = _to_bool(config["train"].get("grad_smooth_grad1", False), default=False)
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

    train_cfg = config["train"]
    parameters = {
        "model_mode": model_mode,
        "use_gradient_data": use_gradient_data,
        "use_time_feature": use_time_feature,
        "use_cycle_loss": use_cycle_loss,
        "enable_imu_cycle_loss": enable_imu_cycle_loss,
        "enable_pressure_cycle_loss": enable_pressure_cycle_loss,
        "freeze_pretrained_cycle_nets": freeze_pretrained_cycle_nets,
        "pose_loss_weight_2d": pose_loss_weight_2d,
        "pose_loss_weight_3d": pose_loss_weight_3d,
        "imu_cycle_loss_weight": imu_cycle_loss_weight,
        "pressure_cycle_loss_weight": pressure_cycle_loss_weight,
        "use_lower_leg_angles_for_accelnet": bool(use_lower_leg_angles_for_accelnet and model_mode == "soleformer"),
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        shuffle=True,
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

    if cycle_training_active:
        accel_input_dim = parameters["output_dim"]
        if parameters["use_lower_leg_angles_for_accelnet"]:
            # Two feet orientation vectors (ankle->toe), 3 axes each.
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

        # Optional: Pretrain auxiliaries if enabled and no checkpoint provided
        if pretrain_accelnet_enabled and not accelnet_pretrained_path:
            accelnet_save_path = os.path.join(".", "results", "pretrained_aux", "accelnet_pretrained.pt")
            os.makedirs(os.path.dirname(accelnet_save_path), exist_ok=True)
            print("\n" + "="*60)
            print("PRETRAINING AccelNet (pose → 6DoF IMU)...")
            print("="*60)
            pretrain_accelnet(
                accel_net,
                train_loader,
                val_loader,
                num_epochs=pretrain_epochs,
                learning_rate=pretrain_learning_rate,
                save_path=accelnet_save_path,
                device=device,
            )
            accelnet_pretrained_path = accelnet_save_path
            print(f"Using newly pretrained AccelNet from {accelnet_save_path}")

        if pretrain_pressnet_enabled and not pressnet_pretrained_path:
            pressnet_save_path = os.path.join(".", "results", "pretrained_aux", "pressnet_pretrained.pt")
            os.makedirs(os.path.dirname(pressnet_save_path), exist_ok=True)
            print("\n" + "="*60)
            print("PRETRAINING PressNet (pose → foot pressure)...")
            print("="*60)
            pretrain_pressnet(
                press_net,
                train_loader,
                val_loader,
                num_epochs=pretrain_epochs,
                learning_rate=pretrain_learning_rate,
                save_path=pressnet_save_path,
                device=device,
            )
            pressnet_pretrained_path = pressnet_save_path
            print(f"Using newly pretrained PressNet from {pressnet_save_path}")

        if accelnet_pretrained_path:
            if not os.path.isfile(accelnet_pretrained_path):
                raise FileNotFoundError(f"AccelNet checkpoint not found: {accelnet_pretrained_path}")
            accel_ckpt = torch.load(accelnet_pretrained_path, map_location="cpu")
            accel_state = accel_ckpt["model_state_dict"] if isinstance(accel_ckpt, dict) and "model_state_dict" in accel_ckpt else accel_ckpt
            accel_net.load_state_dict(accel_state, strict=True)
            accelnet_loaded = True

        if pressnet_pretrained_path:
            if not os.path.isfile(pressnet_pretrained_path):
                raise FileNotFoundError(f"PressNet checkpoint not found: {pressnet_pretrained_path}")
            press_ckpt = torch.load(pressnet_pretrained_path, map_location="cpu")
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
        "target_column_names": list(target_df.columns),
        "target_position_columns": awinda_target_meta.get("position_columns", []),
        "target_angle_columns": awinda_target_meta.get("angle_columns", []),
        "joint_angles_tab_suffix": awinda_target_meta.get("joint_angles_suffix", "n/a"),
        "target_output_dim": int(target_df.shape[1]),
        "train_use_cycle_loss": bool(cycle_training_active),
        "train_enable_imu_cycle_loss": bool(enable_imu_cycle_loss),
        "train_enable_pressure_cycle_loss": bool(enable_pressure_cycle_loss),
        "train_pose_loss_weight_2d": pose_loss_weight_2d,
        "train_pose_loss_weight_3d": pose_loss_weight_3d,
        "train_imu_cycle_loss_weight": imu_cycle_loss_weight,
        "train_pressure_cycle_loss_weight": pressure_cycle_loss_weight,
        "train_use_lower_leg_angles_for_accelnet": bool(parameters["use_lower_leg_angles_for_accelnet"]),
        "train_use_weight_decay_schedule": bool(parameters["use_weight_decay_schedule"]),
        "train_weight_decay_warmup_epochs": int(parameters["weight_decay_warmup_epochs"]),
        "train_min_weight_decay": float(parameters["min_weight_decay"]),
        "train_freeze_pretrained_cycle_nets": bool(freeze_pretrained_cycle_nets),
        "accelnet_pretrained_path": accelnet_pretrained_path,
        "pressnet_pretrained_path": pressnet_pretrained_path,
        "accelnet_pretrained_loaded": bool(accelnet_loaded),
        "pressnet_pretrained_loaded": bool(pressnet_loaded),
    }

    checkpoint_extra["preprocessing_use_time_feature"] = bool(use_time_feature)
    # Per-segment normalization does not require scaler state to be persisted.

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

    best_ckpt_name = join_nonempty("best_skeleton_model", abl_tag, parameters["model_mode"])
    final_ckpt_name = join_nonempty("final_skeleton_model", abl_tag, parameters["model_mode"])
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
        f"{join_nonempty('Learning_results', abl_tag, parameters['model_mode'])}.csv",
    )

    if loss_history:
        # Build deterministic fieldnames in consistent order
        fieldnames = ["epoch", "train_loss", "val_loss"]
        component_keys = set()
        for row in loss_history:
            for key in row.keys():
                if key not in fieldnames:
                    component_keys.add(key)
        # Sort component keys for deterministic ordering
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
        },
        **checkpoint_extra,
    }
    if cycle_training_active:
        final_checkpoint["accel_net_state_dict"] = accel_net.state_dict()
        final_checkpoint["press_net_state_dict"] = press_net.state_dict()
    torch.save(final_checkpoint, final_ckpt_path)


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
    parser.add_argument("--model_mode", type=str, default=None, choices=["original", "simple_seq2seq", "soleformer"])
    parser.add_argument("--abl_id", type=str, default=None)
    parser.add_argument("--use_time_feature", type=str, default=None)
    parser.add_argument("--use_gradient_data", type=str, default=None)
    parser.add_argument("--use_cycle_loss", type=str, default=None)
    parser.add_argument("--enable_imu_cycle_loss", type=str, default=None)
    parser.add_argument("--enable_pressure_cycle_loss", type=str, default=None)
    parser.add_argument("--freeze_pretrained_cycle_nets", type=str, default=None)
    parser.add_argument("--pose_loss_weight_2d", type=float, default=None)
    parser.add_argument("--pose_loss_weight_3d", type=float, default=None)
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
    parser.add_argument("--awinda_tabs_dir", type=str, default=None)
    parser.add_argument("--include_target_positions", type=str, default=None)
    parser.add_argument("--include_target_joint_angles", type=str, default=None)
    parser.add_argument("--joint_angles_tab_suffix", type=str, default=None)

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
    parser.add_argument("--soleformer_use_weight_decay_schedule", type=str, default=None)
    parser.add_argument("--soleformer_weight_decay_warmup_epochs", type=int, default=None)
    parser.add_argument("--soleformer_min_weight_decay", type=float, default=None)

    return parser
