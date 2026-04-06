# File for building deep learning models
import datetime
import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]  # type: ignore


class Transformer_Encoder(nn.Module):
    # Original sequence-to-one model.
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1):
        super().__init__()
        self.model_mode = "original"

        self.num_joints = num_joints
        self.num_dims = num_dims

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.positional_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_encoder_layers,
        )

        self.output_decoder_l1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_decoder_l2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_joints * num_dims),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.positional_encoder(features)
        encoded = self.transformer_encoder(features)
        last_frame = encoded[:, -1, :]
        decoded = self.output_decoder_l1(last_frame)
        decoded = decoded + last_frame
        output = self.output_decoder_l2(decoded)
        return output


class Transformer_Encoder_Seq2Seq(nn.Module):
    # Simple sequence-to-sequence model.
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1):
        super().__init__()
        self.model_mode = "simple_seq2seq"

        self.num_joints = num_joints
        self.num_dims = num_dims

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.positional_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_encoder_layers,
        )

        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_joints * num_dims),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.positional_encoder(features)
        encoded = self.transformer_encoder(features)
        output = self.output_decoder(encoded)
        return output


class AccelNet(nn.Module):
    """Pre-trained MLP for cycle consistency: predicts 6DoF acceleration from pose joints."""
    def __init__(self, input_dim, output_dim=12, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Input: pose features (batch, seq_len, pose_dim)"""
        return self.network(x)


class PressNet(nn.Module):
    """Pre-trained ResNet-based MLP for cycle consistency: predicts foot pressure from pose."""
    def __init__(self, input_dim, output_dim=32, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Input: pose features (batch, seq_len, pose_dim) or flattened"""
        out = self.fc_in(x)
        out = out + self.fc_res(out)
        out = self.fc_out(out)
        return out


class GraphPressureNet(nn.Module):
    """Graph Neural Network for pressure sensor feature extraction."""
    def __init__(self, pressure_dim, d_model, dropout=0.1):
        super().__init__()
        # Treat 16 pressure sensors as nodes in a graph
        self.num_sensors = 16
        self.pressure_dim = pressure_dim
        
        # Initial feature projection per sensor
        self.sensor_projection = nn.Sequential(
            nn.Linear(2, d_model),  # 2 sensors per foot (left, right)
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Graph attention layers
        self.graph_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, x):
        """Input: (batch, seq_len, 32) - 16 sensors * 2 feet"""
        batch_size, seq_len, _ = x.shape
        
        # Reshape to (batch, seq_len, 16, 2)
        x = x.view(batch_size, seq_len, self.num_sensors, 2)
        
        # Project each sensor: (batch, seq_len, 16, d_model)
        x = self.sensor_projection(x)
        
        # Flatten for attention: (batch*seq_len, 16, d_model)
        x_flat = x.view(batch_size * seq_len, self.num_sensors, -1)
        
        # Apply graph attention (self-attention across sensors)
        x_attn, _ = self.graph_attention(x_flat, x_flat, x_flat)
        
        # Project output
        x_out = self.output_projection(x_attn)
        
        # Global pooling across sensors: (batch*seq_len, d_model)
        x_pooled = x_out.mean(dim=1)
        
        # Reshape back: (batch, seq_len, d_model)
        return x_pooled.view(batch_size, seq_len, -1)


class SoleFormer(nn.Module):
    """SoleFormer: Two-stream Transformer with cross-attention for pose estimation from insole sensors.
    
    Architecture:
    - Pressure stream: Graph Neural Network extracting spatial relationships between pressure sensors
    - IMU stream: MLP extracting features from 6DoF IMU data
    - Cross-attention mechanism: Learning relationships between pressure and IMU inputs
    - Cycle consistency losses: Enforcing physical constraints through AccelNet and PressNet
    """
    def __init__(
        self,
        pressure_dim,
        imu_dim,
        d_model,
        nhead,
        num_encoder_layers,
        output_dim,
        dropout=0.1,
        use_graph_pressure=True,
    ):
        super().__init__()
        self.model_mode = "soleformer"

        self.pressure_dim = int(pressure_dim)
        self.imu_dim = int(imu_dim)
        self.output_dim = int(output_dim)
        self.num_joints = self.output_dim // 3
        self.num_dims = 3
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.use_graph_pressure = use_graph_pressure

        # Stream 1: Pressure feature extraction using Graph Neural Network
        if use_graph_pressure:
            self.pressure_feature_extractor = GraphPressureNet(
                self.pressure_dim, d_model, dropout=dropout
            )
        else:
            self.pressure_feature_extractor = nn.Sequential(
                nn.Linear(self.pressure_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

        # Stream 2: IMU feature extraction
        self.imu_feature_extractor = nn.Sequential(
            nn.Linear(self.imu_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding for temporal information
        self.positional_encoder_pressure = PositionalEncoding(d_model)
        self.positional_encoder_imu = PositionalEncoding(d_model)

        # Two-stream Transformer: Self-attention + Cross-attention layers
        self.pressure_self_layers = nn.ModuleList()
        self.imu_self_layers = nn.ModuleList()
        self.pressure_to_imu_cross_layers = nn.ModuleList()  # Pressure queries, IMU keys/values
        self.imu_to_pressure_cross_layers = nn.ModuleList()  # IMU queries, Pressure keys/values
        self.pressure_norm_layers = nn.ModuleList()
        self.imu_norm_layers = nn.ModuleList()
        self.pressure_cross_norm_layers = nn.ModuleList()
        self.imu_cross_norm_layers = nn.ModuleList()

        for _ in range(num_encoder_layers):
            # Self-attention for each stream
            self.pressure_self_layers.append(
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            )
            self.imu_self_layers.append(
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            )
            
            # Cross-attention: pressure queries attend to IMU keys/values
            self.pressure_to_imu_cross_layers.append(
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            )
            
            # Cross-attention: IMU queries attend to pressure keys/values
            self.imu_to_pressure_cross_layers.append(
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            )
            
            # Layer normalization
            self.pressure_norm_layers.append(nn.LayerNorm(d_model))
            self.imu_norm_layers.append(nn.LayerNorm(d_model))
            self.pressure_cross_norm_layers.append(nn.LayerNorm(d_model))
            self.imu_cross_norm_layers.append(nn.LayerNorm(d_model))

        # Fusion decoder
        self.fusion_decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.output_dim),
        )

    def forward(self, x):
        """Forward pass for SoleFormer.
        
        Args:
            x: (batch, seq_len, pressure_dim + imu_dim)
        
        Returns:
            output: (batch, seq_len, output_dim)
        """
        # Split pressure and IMU inputs
        pressure_x = x[..., : self.pressure_dim]
        imu_x = x[..., self.pressure_dim : self.pressure_dim + self.imu_dim]

        # Feature extraction
        pressure_feat = self.pressure_feature_extractor(pressure_x)
        imu_feat = self.imu_feature_extractor(imu_x)

        # Positional encoding
        pressure_feat = self.positional_encoder_pressure(pressure_feat)
        imu_feat = self.positional_encoder_imu(imu_feat)

        # Two-stream transformer with cross-attention
        for i in range(self.num_encoder_layers):
            # Self-attention within each stream
            pressure_self, _ = self.pressure_self_layers[i](
                pressure_feat, pressure_feat, pressure_feat
            )
            imu_self, _ = self.imu_self_layers[i](imu_feat, imu_feat, imu_feat)

            # Add & normalize (residual connection)
            pressure_feat = self.pressure_norm_layers[i](pressure_feat + pressure_self)
            imu_feat = self.imu_norm_layers[i](imu_feat + imu_self)

            # Cross-attention between streams
            # Pressure stream attends to IMU information
            pressure_cross, _ = self.pressure_to_imu_cross_layers[i](
                pressure_feat, imu_feat, imu_feat
            )
            # IMU stream attends to pressure information
            imu_cross, _ = self.imu_to_pressure_cross_layers[i](
                imu_feat, pressure_feat, pressure_feat
            )

            # Add & normalize
            pressure_feat = self.pressure_cross_norm_layers[i](pressure_feat + pressure_cross)
            imu_feat = self.imu_cross_norm_layers[i](imu_feat + imu_cross)

        # Fusion and output
        fused = torch.cat([pressure_feat, imu_feat], dim=-1)
        output = self.fusion_decoder(fused)
        
        return output


class DoubleCycleConsistencyLoss(nn.Module):
    """Double-cycle consistency loss for SoleFormer.
    
    Enforces physical constraints:
    1. IMU Cycle: pose -> AccelNet -> predicted acceleration (should match input IMU)
    2. Pressure Cycle: pose -> PressNet -> predicted pressure (should match input pressure)
    """
    def __init__(
        self,
        accel_net,
        press_net,
        weight_pose=1.0,
        weight_imu_cycle=0.5,
        weight_pressure_cycle=0.5,
        weight_2d_loss=1.0,
        weight_3d_loss=1.0,
        enable_imu_cycle=True,
        enable_pressure_cycle=True,
        use_lower_leg_angles_for_accelnet=False,
        accelnet_foot_indices=None,
    ):
        super().__init__()
        self.accel_net = accel_net
        self.press_net = press_net
        self.weight_pose = weight_pose
        self.weight_imu_cycle = weight_imu_cycle
        self.weight_pressure_cycle = weight_pressure_cycle
        self.weight_2d_loss = weight_2d_loss
        self.weight_3d_loss = weight_3d_loss
        self.enable_imu_cycle = enable_imu_cycle
        self.enable_pressure_cycle = enable_pressure_cycle
        self.use_lower_leg_angles_for_accelnet = bool(use_lower_leg_angles_for_accelnet)
        self.accelnet_foot_indices = accelnet_foot_indices

    @staticmethod
    def extract_foot_orientation_features(pose, foot_indices=None):
        """Extract ankle->toe orientation vectors for both feet.

        Returns a tensor of shape (batch, seq_len, 6) when possible.
        Returns None if required joints are unavailable.
        """
        if pose.dim() != 3 or pose.shape[-1] % 3 != 0:
            return None

        n_joints = pose.shape[-1] // 3
        if foot_indices is None:
            if n_joints >= 23:
                # Awinda 23-joint targets: RightFoot/RightToe and LeftFoot/LeftToe.
                right_ankle, right_toe, left_ankle, left_toe = 17, 18, 21, 22
            else:
                return None
        else:
            right_ankle, right_toe, left_ankle, left_toe = foot_indices

        if max(right_ankle, right_toe, left_ankle, left_toe) >= n_joints:
            return None

        joints = pose.view(*pose.shape[:-1], n_joints, 3)
        right_vec = joints[:, :, right_toe, :] - joints[:, :, right_ankle, :]
        left_vec = joints[:, :, left_toe, :] - joints[:, :, left_ankle, :]

        right_vec = F.normalize(right_vec, dim=-1, eps=1e-6)
        left_vec = F.normalize(left_vec, dim=-1, eps=1e-6)
        return torch.cat([right_vec, left_vec], dim=-1)

    def forward(self, pred_pose, target_pose, input_imu, input_pressure):
        """Compute double-cycle consistency loss.
        
        Args:
            pred_pose: (batch, seq_len, pose_dim) - Predicted pose from SoleFormer
            target_pose: (batch, seq_len, pose_dim) - Ground truth pose
            input_imu: (batch, seq_len, imu_dim) - Input IMU data
            input_pressure: (batch, seq_len, pressure_dim) - Input pressure data
        
        Returns:
            total_loss: scalar
        """
        pose_2d_loss = pred_pose.new_tensor(0.0)
        if pred_pose.shape[-1] % 3 == 0:
            pred_pose_xyz = pred_pose.view(*pred_pose.shape[:-1], -1, 3)
            target_pose_xyz = target_pose.view(*target_pose.shape[:-1], -1, 3)
            pose_2d_loss = F.mse_loss(pred_pose_xyz[..., :2], target_pose_xyz[..., :2])
            pose_3d_loss = F.mse_loss(pred_pose_xyz, target_pose_xyz)
            pose_loss = self.weight_2d_loss * pose_2d_loss + self.weight_3d_loss * pose_3d_loss
        else:
            pose_3d_loss = F.mse_loss(pred_pose, target_pose)
            pose_loss = pose_3d_loss

        imu_cycle_loss = pred_pose.new_tensor(0.0)
        if self.enable_imu_cycle:
            accel_input = pred_pose
            if self.use_lower_leg_angles_for_accelnet:
                foot_orient = self.extract_foot_orientation_features(
                    pred_pose,
                    foot_indices=self.accelnet_foot_indices,
                )
                if foot_orient is not None:
                    accel_input = foot_orient

            pred_imu = self.accel_net(accel_input)
            if pred_imu.shape != input_imu.shape:
                shared_t = min(pred_imu.shape[1], input_imu.shape[1])
                shared_c = min(pred_imu.shape[2], input_imu.shape[2])
                if shared_t <= 0 or shared_c <= 0:
                    raise ValueError(
                        f"AccelNet output shape {tuple(pred_imu.shape)} does not match input IMU shape {tuple(input_imu.shape)}"
                    )
                pred_imu = pred_imu[:, :shared_t, :shared_c]
                input_imu = input_imu[:, :shared_t, :shared_c]
            imu_cycle_loss = F.mse_loss(pred_imu, input_imu)

        pressure_cycle_loss = pred_pose.new_tensor(0.0)
        if self.enable_pressure_cycle:
            pred_pressure = self.press_net(pred_pose)
            if pred_pressure.shape != input_pressure.shape:
                shared_t = min(pred_pressure.shape[1], input_pressure.shape[1])
                shared_c = min(pred_pressure.shape[2], input_pressure.shape[2])
                if shared_t <= 0 or shared_c <= 0:
                    raise ValueError(
                        f"PressNet output shape {tuple(pred_pressure.shape)} does not match input pressure shape {tuple(input_pressure.shape)}"
                    )
                pred_pressure = pred_pressure[:, :shared_t, :shared_c]
                input_pressure = input_pressure[:, :shared_t, :shared_c]
            pressure_cycle_loss = F.mse_loss(pred_pressure, input_pressure)

        total_loss = self.weight_pose * pose_loss
        total_loss = total_loss + self.weight_imu_cycle * imu_cycle_loss
        total_loss = total_loss + self.weight_pressure_cycle * pressure_cycle_loss

        return total_loss, {
            "pose_loss": pose_loss.item(),
            "pose_2d_loss": pose_2d_loss.item(),
            "pose_3d_loss": pose_3d_loss.item(),
            "imu_cycle_loss": imu_cycle_loss.item(),
            "pressure_cycle_loss": pressure_cycle_loss.item(),
        }


class Skeleton_Loss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


def _build_model_config(model):
    model_config = {
        "num_joints": getattr(model, "num_joints", None),
        "model_mode": getattr(model, "model_mode", None),
    }
    if hasattr(model, "feature_extractor"):
        model_config["input_dim"] = model.feature_extractor[0].in_features
        model_config["d_model"] = model.feature_extractor[0].out_features
        model_config["nhead"] = model.transformer_encoder.layers[0].self_attn.num_heads
        model_config["num_encoder_layers"] = len(model.transformer_encoder.layers)
    elif hasattr(model, "pressure_feature_extractor") and hasattr(model, "imu_feature_extractor"):
        model_config["pressure_dim"] = model.pressure_dim
        model_config["imu_dim"] = model.imu_dim
        model_config["input_dim"] = model.pressure_dim + model.imu_dim
        model_config["d_model"] = model.d_model
        model_config["nhead"] = model.pressure_self_layers[0].num_heads
        model_config["num_encoder_layers"] = len(model.pressure_self_layers)
        model_config["output_dim"] = model.output_dim
    return model_config


def train_mse(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    save_path,
    device,
    checkpoint_extra=None,
    wd_scheduler=None,
):
    best_val_loss = float("inf")
    history = []

    model_config = _build_model_config(model)

    start_time = time.time()
    pre_time = start_time
    elaps_time_total_s = 0.0
    now_time = datetime.datetime.now()
    print(f"\n[train started at {now_time.strftime('%H:%M')}]")

    for epoch in range(num_epochs):
        if wd_scheduler is not None:
            wd_scheduler.step(epoch)

        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", leave=False)
        for pressure, skeleton in train_pbar:
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)

            optimizer.zero_grad()
            outputs = model(pressure)
            loss = criterion(outputs, skeleton)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} Val", leave=False)
        with torch.no_grad():
            for pressure, skeleton in val_pbar:
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)
                outputs = model(pressure)
                loss = criterion(outputs, skeleton)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

        end_time = time.time()
        epoch_time_total_s = end_time - pre_time
        elaps_time_total_s = end_time - start_time
        epoch_time_m = int(epoch_time_total_s // 60)
        epoch_time_s = int(epoch_time_total_s % 60)
        elaps_time_m = int(elaps_time_total_s // 60)
        elaps_time_s = int(elaps_time_total_s % 60)

        remaining_epochs = num_epochs - (epoch + 1)
        est_remaining_time_s = epoch_time_total_s * remaining_epochs
        est_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_time_s)
        est_finish_str = est_finish_time.strftime("%H:%M")

        print(
            f"------------ Epoch {epoch + 1}/{num_epochs} ------------\n"
            f"Train Loss (MSE): {avg_train_loss:.6f}\n"
            f"Val Loss (MSE): {avg_val_loss:.6f}\n"
            f"LR        : {current_lr:.5f}\n"
            f"WD        : {current_wd:.6f}\n"
            f"Time/epoch: {epoch_time_m}m {epoch_time_s}s | Total: {elaps_time_m}m {elaps_time_s}s\n"
            f"Estimated Finish: {est_finish_str}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
            }
        )

        pre_time = end_time

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "model_config": model_config,
                **(checkpoint_extra or {}),
            }
            torch.save(checkpoint, save_path)
            print(f">> Model saved at epoch {epoch + 1} (val_loss={best_val_loss:.6f})")

    return history


def train_mse_with_cycle(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    save_path,
    device,
    checkpoint_extra=None,
    wd_scheduler=None,
):
    best_val_loss = float("inf")
    history = []
    model_config = _build_model_config(model)

    start_time = time.time()
    pre_time = start_time
    elaps_time_total_s = 0.0
    now_time = datetime.datetime.now()
    print(f"\n[cycle train started at {now_time.strftime('%H:%M')}]")

    for epoch in range(num_epochs):
        if wd_scheduler is not None:
            wd_scheduler.step(epoch)

        model.train()
        train_total = 0.0
        train_pose = 0.0
        train_pose2d = 0.0
        train_pose3d = 0.0
        train_imu_cycle = 0.0
        train_pressure_cycle = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", leave=False)
        for pressure, skeleton in train_pbar:
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)

            optimizer.zero_grad()
            outputs = model(pressure)

            input_pressure = pressure[..., : model.pressure_dim]
            input_imu = pressure[..., model.pressure_dim : model.pressure_dim + model.imu_dim]

            total_loss, loss_items = criterion(outputs, skeleton, input_imu, input_pressure)
            total_loss.backward()
            optimizer.step()

            train_total += float(total_loss.item())
            train_pose += float(loss_items.get("pose_loss", 0.0))
            train_pose2d += float(loss_items.get("pose_2d_loss", 0.0))
            train_pose3d += float(loss_items.get("pose_3d_loss", 0.0))
            train_imu_cycle += float(loss_items.get("imu_cycle_loss", 0.0))
            train_pressure_cycle += float(loss_items.get("pressure_cycle_loss", 0.0))

        model.eval()
        val_total = 0.0
        val_pose = 0.0
        val_pose2d = 0.0
        val_pose3d = 0.0
        val_imu_cycle = 0.0
        val_pressure_cycle = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} Val", leave=False)
        with torch.no_grad():
            for pressure, skeleton in val_pbar:
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)
                outputs = model(pressure)

                input_pressure = pressure[..., : model.pressure_dim]
                input_imu = pressure[..., model.pressure_dim : model.pressure_dim + model.imu_dim]

                total_loss, loss_items = criterion(outputs, skeleton, input_imu, input_pressure)
                val_total += float(total_loss.item())
                val_pose += float(loss_items.get("pose_loss", 0.0))
                val_pose2d += float(loss_items.get("pose_2d_loss", 0.0))
                val_pose3d += float(loss_items.get("pose_3d_loss", 0.0))
                val_imu_cycle += float(loss_items.get("imu_cycle_loss", 0.0))
                val_pressure_cycle += float(loss_items.get("pressure_cycle_loss", 0.0))

        avg_train_total = train_total / len(train_loader)
        avg_train_pose = train_pose / len(train_loader)
        avg_train_pose2d = train_pose2d / len(train_loader)
        avg_train_pose3d = train_pose3d / len(train_loader)
        avg_train_imu = train_imu_cycle / len(train_loader)
        avg_train_press = train_pressure_cycle / len(train_loader)

        avg_val_total = val_total / len(val_loader)
        avg_val_pose = val_pose / len(val_loader)
        avg_val_pose2d = val_pose2d / len(val_loader)
        avg_val_pose3d = val_pose3d / len(val_loader)
        avg_val_imu = val_imu_cycle / len(val_loader)
        avg_val_press = val_pressure_cycle / len(val_loader)

        scheduler.step(avg_val_total)
        current_lr = optimizer.param_groups[0]["lr"]
        current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

        end_time = time.time()
        epoch_time_total_s = end_time - pre_time
        elaps_time_total_s = end_time - start_time
        epoch_time_m = int(epoch_time_total_s // 60)
        epoch_time_s = int(epoch_time_total_s % 60)
        elaps_time_m = int(elaps_time_total_s // 60)
        elaps_time_s = int(elaps_time_total_s % 60)

        remaining_epochs = num_epochs - (epoch + 1)
        est_remaining_time_s = epoch_time_total_s * remaining_epochs
        est_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_time_s)
        est_finish_str = est_finish_time.strftime("%H:%M")

        print(
            f"------------ Epoch {epoch + 1}/{num_epochs} ------------\n"
            f"Train Total: {avg_train_total:.6f} | Val Total: {avg_val_total:.6f}\n"
            f"Train Pose: {avg_train_pose:.6f} (2D={avg_train_pose2d:.6f}, 3D={avg_train_pose3d:.6f})\n"
            f"Train Cycles: IMU={avg_train_imu:.6f}, Pressure={avg_train_press:.6f}\n"
            f"Val Pose: {avg_val_pose:.6f} (2D={avg_val_pose2d:.6f}, 3D={avg_val_pose3d:.6f})\n"
            f"Val Cycles: IMU={avg_val_imu:.6f}, Pressure={avg_val_press:.6f}\n"
            f"LR        : {current_lr:.5f}\n"
            f"WD        : {current_wd:.6f}\n"
            f"Time/epoch: {epoch_time_m}m {epoch_time_s}s | Total: {elaps_time_m}m {elaps_time_s}s\n"
            f"Estimated Finish: {est_finish_str}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_total),
                "val_loss": float(avg_val_total),
                "train_pose_loss": float(avg_train_pose),
                "val_pose_loss": float(avg_val_pose),
                "train_pose_2d_loss": float(avg_train_pose2d),
                "val_pose_2d_loss": float(avg_val_pose2d),
                "train_pose_3d_loss": float(avg_train_pose3d),
                "val_pose_3d_loss": float(avg_val_pose3d),
                "train_imu_cycle_loss": float(avg_train_imu),
                "val_imu_cycle_loss": float(avg_val_imu),
                "train_pressure_cycle_loss": float(avg_train_press),
                "val_pressure_cycle_loss": float(avg_val_press),
            }
        )

        pre_time = end_time

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "model_config": model_config,
                **(checkpoint_extra or {}),
            }
            torch.save(checkpoint, save_path)
            print(f">> Model saved at epoch {epoch + 1} (val_total={best_val_loss:.6f})")

    return history


def pretrain_accelnet(
    accelnet,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    save_path=None,
    device="cpu",
):
    """
    Pretrain AccelNet: pose → 6DoF IMU (acceleration + gyroscope).
    
    Args:
        accelnet: AccelNet instance
        train_loader: DataLoader with (pressure, skeleton) tuples
        val_loader: DataLoader with (pressure, skeleton) tuples
        num_epochs: Training epochs
        learning_rate: Optimizer learning rate
        save_path: Path to save best checkpoint
        device: 'cpu' or 'cuda'
    
    Note:
        Expects train_loader batches: (pressure_batch, skeleton_batch)
        where pressure_batch shape is (batch, seq_len, pressure_dim + imu_dim + ...).
        Extracts IMU from columns [pressure_dim : pressure_dim + imu_dim].
        Flattens skeleton (batch, seq_len, pose_dim) as input to AccelNet.
    """
    if save_path is None:
        save_path = os.path.join(".", "results", "pretrained_aux", "accelnet_pretrained.pt")
    
    optimizer = torch.optim.Adam(accelnet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    accelnet.to(device)
    best_val_loss = float("inf")
    _printed_dim_warning = False
    
    start_time = time.time()
    pre_time = start_time
    elaps_time_total_s = 0.0
    now_time = datetime.datetime.now()
    print(f"\n[AccelNet pretraining started at {now_time.strftime('%H:%M')}]")
    
    for epoch in range(num_epochs):
        accelnet.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"AccelNet Epoch {epoch + 1} Train", leave=False)
        for pressure, skeleton in train_pbar:
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)
            
            # Extract IMU from pressure concatenation
            # Assume: pressure_batch = [pressure_channels | imu_dim | possibly_more]
            imu_dim = int(getattr(accelnet, "_imu_dim", accelnet.network[-1].out_features))
            pressure_dim = int(getattr(accelnet, "_pressure_dim", pressure.shape[2] - imu_dim))
            imu_start = int(getattr(accelnet, "_imu_start", pressure_dim))
            imu_end = imu_start + imu_dim
            target_imu = pressure[:, :, imu_start:imu_end]  # (batch, seq_len, imu_dim)
            
            # Flatten skeleton as input: (batch, seq_len, pose_dim) → (batch, seq_len, pose_dim)
            input_pose = skeleton  # skeleton is already (batch, seq_len, pose_dim)
            if bool(getattr(accelnet, "_use_lower_leg_angles_for_accelnet", False)):
                foot_orient = DoubleCycleConsistencyLoss.extract_foot_orientation_features(
                    input_pose,
                    foot_indices=getattr(accelnet, "_foot_orientation_indices", None),
                )
                if foot_orient is not None:
                    input_pose = foot_orient
            
            optimizer.zero_grad()
            pred_imu = accelnet(input_pose)  # (batch, seq_len, imu_dim)
            if pred_imu.shape[-1] != target_imu.shape[-1]:
                shared_dim = int(min(pred_imu.shape[-1], target_imu.shape[-1]))
                if shared_dim <= 0:
                    raise ValueError(
                        f"Invalid IMU dimensions for pretraining: pred={tuple(pred_imu.shape)}, "
                        f"target={tuple(target_imu.shape)}"
                    )
                if not _printed_dim_warning:
                    print(
                        "[AccelNet pretrain] Channel mismatch detected; "
                        f"aligning from pred={pred_imu.shape[-1]} and target={target_imu.shape[-1]} "
                        f"to shared_dim={shared_dim}."
                    )
                    _printed_dim_warning = True
                pred_imu = pred_imu[..., :shared_dim]
                target_imu = target_imu[..., :shared_dim]
            loss = criterion(pred_imu, target_imu)
            loss.backward()
            optimizer.step()
            
            train_loss += float(loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        accelnet.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"AccelNet Epoch {epoch + 1} Val", leave=False)
        with torch.no_grad():
            for pressure, skeleton in val_pbar:
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)
                
                imu_dim = int(getattr(accelnet, "_imu_dim", accelnet.network[-1].out_features))
                pressure_dim = int(getattr(accelnet, "_pressure_dim", pressure.shape[2] - imu_dim))
                imu_start = int(getattr(accelnet, "_imu_start", pressure_dim))
                imu_end = imu_start + imu_dim
                target_imu = pressure[:, :, imu_start:imu_end]
                input_pose = skeleton
                if bool(getattr(accelnet, "_use_lower_leg_angles_for_accelnet", False)):
                    foot_orient = DoubleCycleConsistencyLoss.extract_foot_orientation_features(
                        input_pose,
                        foot_indices=getattr(accelnet, "_foot_orientation_indices", None),
                    )
                    if foot_orient is not None:
                        input_pose = foot_orient
                
                pred_imu = accelnet(input_pose)
                if pred_imu.shape[-1] != target_imu.shape[-1]:
                    shared_dim = int(min(pred_imu.shape[-1], target_imu.shape[-1]))
                    if shared_dim <= 0:
                        raise ValueError(
                            f"Invalid IMU dimensions for validation: pred={tuple(pred_imu.shape)}, "
                            f"target={tuple(target_imu.shape)}"
                        )
                    pred_imu = pred_imu[..., :shared_dim]
                    target_imu = target_imu[..., :shared_dim]
                loss = criterion(pred_imu, target_imu)
                val_loss += float(loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        end_time = time.time()
        epoch_time_total_s = end_time - pre_time
        elaps_time_total_s = end_time - start_time
        epoch_time_m = int(epoch_time_total_s // 60)
        epoch_time_s = int(epoch_time_total_s % 60)
        elaps_time_m = int(elaps_time_total_s // 60)
        elaps_time_s = int(elaps_time_total_s % 60)
        
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.5f} | Time: {epoch_time_m}m {epoch_time_s}s"
        )
        
        pre_time = end_time
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"model_state_dict": accelnet.state_dict(), "best_val_loss": best_val_loss}, save_path)
            print(f">> AccelNet saved at epoch {epoch + 1} (val_loss={best_val_loss:.6f})")
    
    total_time_m = int(elaps_time_total_s // 60)
    total_time_s = int(elaps_time_total_s % 60)
    print(f"\n[AccelNet pretraining completed in {total_time_m}m {total_time_s}s, best val loss={best_val_loss:.6f}]\n")


def pretrain_pressnet(
    pressnet,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    save_path=None,
    device="cpu",
):
    """
    Pretrain PressNet: pose → foot pressure (32 channels).
    
    Args:
        pressnet: PressNet instance
        train_loader: DataLoader with (pressure, skeleton) tuples
        val_loader: DataLoader with (pressure, skeleton) tuples
        num_epochs: Training epochs
        learning_rate: Optimizer learning rate
        save_path: Path to save best checkpoint
        device: 'cpu' or 'cuda'
    
    Note:
        Expects train_loader batches: (pressure_batch, skeleton_batch)
        where pressure_batch shape is (batch, seq_len, pressure_dim + imu_dim + ...).
        Extracts pressure from first <pressure_dim> columns.
        Flattens skeleton (batch, seq_len, pose_dim) as input to PressNet.
    """
    if save_path is None:
        save_path = os.path.join(".", "results", "pretrained_aux", "pressnet_pretrained.pt")
    
    optimizer = torch.optim.Adam(pressnet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    pressnet.to(device)
    best_val_loss = float("inf")
    
    start_time = time.time()
    pre_time = start_time
    elaps_time_total_s = 0.0
    now_time = datetime.datetime.now()
    print(f"\n[PressNet pretraining started at {now_time.strftime('%H:%M')}]")
    
    for epoch in range(num_epochs):
        pressnet.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"PressNet Epoch {epoch + 1} Train", leave=False)
        for pressure, skeleton in train_pbar:
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)
            
            # Extract pressure from first pressure_dim columns
            # Assume: pressure_batch = [pressure_channels | imu_dim | possibly_more]
            pressure_dim = int(getattr(pressnet, "_pressure_dim", pressnet.fc_out.out_features))
            target_pressure = pressure[:, :, :pressure_dim]  # (batch, seq_len, pressure_dim)
            
            # Flatten skeleton as input
            input_pose = skeleton  # (batch, seq_len, pose_dim)
            
            optimizer.zero_grad()
            pred_pressure = pressnet(input_pose)  # (batch, seq_len, pressure_dim)
            loss = criterion(pred_pressure, target_pressure)
            loss.backward()
            optimizer.step()
            
            train_loss += float(loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        pressnet.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"PressNet Epoch {epoch + 1} Val", leave=False)
        with torch.no_grad():
            for pressure, skeleton in val_pbar:
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)
                
                pressure_dim = int(getattr(pressnet, "_pressure_dim", pressnet.fc_out.out_features))
                target_pressure = pressure[:, :, :pressure_dim]
                input_pose = skeleton
                
                pred_pressure = pressnet(input_pose)
                loss = criterion(pred_pressure, target_pressure)
                val_loss += float(loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        end_time = time.time()
        epoch_time_total_s = end_time - pre_time
        elaps_time_total_s = end_time - start_time
        epoch_time_m = int(epoch_time_total_s // 60)
        epoch_time_s = int(epoch_time_total_s % 60)
        elaps_time_m = int(elaps_time_total_s // 60)
        elaps_time_s = int(elaps_time_total_s % 60)
        
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.5f} | Time: {epoch_time_m}m {epoch_time_s}s"
        )
        
        pre_time = end_time
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"model_state_dict": pressnet.state_dict(), "best_val_loss": best_val_loss}, save_path)
            print(f">> PressNet saved at epoch {epoch + 1} (val_loss={best_val_loss:.6f})")
    
    total_time_m = int(elaps_time_total_s // 60)
    total_time_s = int(elaps_time_total_s % 60)
    print(f"\n[PressNet pretraining completed in {total_time_m}m {total_time_s}s, best val loss={best_val_loss:.6f}]\n")


def save_predictions(predictions, model, frame_indices=None, output_stem=None, column_names=None):
    if column_names is not None:
        if len(column_names) != predictions.shape[1]:
            raise ValueError(
                f"Length mismatch: column_names has {len(column_names)} columns but "
                f"predictions has {predictions.shape[1]} columns"
            )
        columns = list(column_names)
    elif predictions.shape[1] % 3 == 0:
        num_joints = predictions.shape[1] // 3
        columns = []
        for i in range(num_joints):
            columns.extend([f"X.{i}", f"Y.{i}", f"Z.{i}"])
    else:
        columns = [f"target_{i}" for i in range(predictions.shape[1])]

    output_name = f"Predicted_skeleton_{output_stem}" if output_stem else "Predicted_skeleton"
    output_file = os.path.join(".", "results", "output", f"{output_name}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_predictions = pd.DataFrame(predictions, columns=columns)
    if frame_indices is not None:
        if len(frame_indices) != len(df_predictions):
            raise ValueError(
                f"Length mismatch: frame_indices has {len(frame_indices)} rows but "
                f"predictions has {len(df_predictions)} rows"
            )
        df_predictions.insert(0, "Frame", frame_indices)
    df_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
