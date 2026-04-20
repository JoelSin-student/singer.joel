# Auxiliary Network Pretraining Guide

Only for SoleFormer model mode.

## Overview

**AccelNet** and **PressNet** are optional auxiliary networks that map skeleton (pose) to IMU (6DoF acceleration/gyro) and foot pressure respectively. They can be pretrained independently before being used as frozen regularizers in the main SoleFormer training loop.

## Architecture

- **AccelNet**: FC layer → pose features → 6DoF IMU output
- **PressNet**: FC layers → pose features → 32-channel foot pressure output

Both can be:
1. **Ignored**: Cycle loss disabled (`use_cycle_loss: false`)
2. **Trained from scratch**: Randomly initialized, trained jointly with SoleFormer
3. **Pretrained separately**: Pretrained on pose→auxiliary task, then used as frozen regularizers

## Pretraining Strategy

Those strategies can be selected in main.ipynb with the different ablation commands, expect for hyperparameters like the number of epochs. Those hyperparameters can be defined by hand in the commandlines, or the sources\config\transformer_encoder .yaml files.

### Strategy 1: Pretrain from Scratch (Recommended for new data)

**Config** (`train.yaml`):
```yaml
use_cycle_loss: true
pretrain_accelnet: true      # Pretrain AccelNet
pretrain_pressnet: true      # Pretrain PressNet
pretrain_epochs: 30          # Sufficient epochs. Increase or decrease in function of pretraining curves
pretrain_learning_rate: 0.001
freeze_pretrained_cycle_nets: true  # Freeze after main training starts
```

**CLI override**:
```bash
python train.py \
  --pretrain_accelnet=true \
  --pretrain_pressnet=true \
  --pretrain_epochs=50 \
  --pretrain_learning_rate=0.0005
```

**Workflow**:
1. AccelNet trained for 30 epochs on pose→IMU task (MSE loss)
   - Saves: `results/pretrained_aux/accelnet_pretrained.pt`
2. PressNet trained for 30 epochs on pose→pressure task (MSE loss)
   - Saves: `results/pretrained_aux/pressnet_pretrained.pt`
3. Main SoleFormer training begins with frozen cycle nets
4. Final checkpoint includes both accel_net and press_net state dicts

### Strategy 2: Load Pretrained Checkpoints

**If you already have trained AccelNet/PressNet**:
```yaml
use_cycle_loss: true
pretrain_accelnet: false     # Skip pretraining
pretrain_pressnet: false     # Skip pretraining
accelnet_pretrained_path: "results/pretrained_aux/accelnet_my_model.pt"
pressnet_pretrained_path: "results/pretrained_aux/pressnet_my_model.pt"
freeze_pretrained_cycle_nets: true
```

**CLI override**:
```bash
python train.py \
  --accelnet_pretrained_path="results/pretrained_aux/accelnet_my_model.pt" \
  --pressnet_pretrained_path="results/pretrained_aux/pressnet_my_model.pt"
```

### Strategy 3: Joint Training (Trainable Auxiliaries)

**If you want auxiliary nets to learn jointly with SoleFormer**:
```yaml
use_cycle_loss: true
pretrain_accelnet: false
pretrain_pressnet: false
freeze_pretrained_cycle_nets: false  # Make them trainable
```

**Effect**: AccelNet and PressNet gradients propagate through the cycle losses; slower but may adapt better to SoleFormer.

### Strategy 4: Cycle Loss Disabled

```yaml
use_cycle_loss: false
```

**Effect**: Only skeleton prediction loss (no IMU/pressure regularization). Fastest training.

## Pretraining Function Details

### `pretrain_accelnet(accelnet, train_loader, val_loader, ...)`

Trains AccelNet to map pose → 6DoF IMU via MSE loss.

- **Input**: Skeleton sequence from data loader (batch, seq_len, pose_dim)
- **Target**: IMU extracted from middle 6 channels of pressure concatenation
- **Loss**: L2 distance between predicted and ground-truth IMU
- **Output**: Best checkpoint saved to `save_path`

### `pretrain_pressnet(pressnet, train_loader, val_loader, ...)`

Trains PressNet to map pose → 32-ch foot pressure via MSE loss.

- **Input**: Skeleton sequence from data loader (batch, seq_len, pose_dim)
- **Target**: Pressure (first `pressure_dim` channels of concatenated data)
- **Loss**: L2 distance between predicted and ground-truth pressure
- **Output**: Best checkpoint saved to `save_path`

Both use:
- **Optimizer**: Adam (lr=0.001 default)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Val metric**: Avg MSE loss

## Workflow in `train.py`

```python
# 1. Read config
pretrain_accelnet = config["train"].get("pretrain_accelnet", False)
pretrain_pressnet = config["train"].get("pretrain_pressnet", False)
pretrain_epochs = config["train"].get("pretrain_epochs", 30)
pretrain_learning_rate = config["train"].get("pretrain_learning_rate", 0.001)

# 2. Create auxiliary nets
accel_net = AccelNet(...)
press_net = PressNet(...)

# 3. Optionally pretrain (if enabled AND no checkpoint path provided)
if pretrain_accelnet and not accelnet_pretrained_path:
    pretrain_accelnet(accel_net, train_loader, val_loader, ...)
    # Auto-saves to results/pretrained_aux/accelnet_pretrained.pt

if pretrain_pressnet and not pressnet_pretrained_path:
    pretrain_pressnet(press_net, train_loader, val_loader, ...)
    # Auto-saves to results/pretrained_aux/pressnet_pretrained.pt

# 4. Load from checkpoint if provided
if accelnet_pretrained_path:
    accel_net.load_state_dict(...)

if pressnet_pretrained_path:
    press_net.load_state_dict(...)

# 5. Optionally freeze
if freeze_pretrained_cycle_nets:
    accel_net.eval()
    press_net.requires_grad_(False)

# 6. Main SoleFormer training with cycle losses
train_mse_with_cycle(model, train_loader, val_loader, 
                     DoubleCycleConsistencyLoss(...), ...)
```

## Expected Output

### Pretraining Phase
```
============================================================
PRETRAINING AccelNet (pose → 6DoF IMU)...
============================================================
AccelNet Epoch 1 Train | Train Loss: 0.234567 | Val Loss: 0.201234 | LR: 0.001000 | Time: 0m 12s
AccelNet Epoch 2 Train | Train Loss: 0.198765 | Val Loss: 0.178901 | LR: 0.001000 | Time: 0m 11s
...
>> AccelNet saved at epoch 28 (val_loss=0.045678)

[AccelNet pretraining completed in 5m 42s, best val loss=0.045678]

============================================================
PRETRAINING PressNet (pose → foot pressure)...
============================================================
PressNet Epoch 1 Train | Train Loss: 0.567890 | Val Loss: 0.523456 | LR: 0.001000 | Time: 0m 15s
...
>> PressNet saved at epoch 24 (val_loss=0.089123)

[PressNet pretraining completed in 6m 18s, best val loss=0.089123]

Cycle training enabled. AccelNet loaded=True, PressNet loaded=True, cycle_nets_frozen=True.
```

### Main Training Phase
```
------------ Epoch 1/100 ------------
Train Total: 0.123456 | Val Total: 0.234567
Train Pose: 0.090123 (2D=0.045061, 3D=0.045062)
Train Cycles: IMU=0.012345, Pressure=0.020988
Val Pose: 0.156789 (2D=0.078394, 3D=0.078395)
Val Cycles: IMU=0.018901, Pressure=0.035234
LR        : 0.00050
Time/epoch: 1m 23s | Total: 1m 23s
...
```

## Tips

1. **Start with defaults**: `pretrain_accelnet=true, pretrain_pressnet=true, freeze_pretrained_cycle_nets=true`
   - Stabilizes early training, helps generalization

2. **Increase pretraining epochs** if auxiliary nets are underfitting:
   - Check val loss plateaus before main training starts
   - Typical range: 10-50 epochs depending on data size and complexity

3. **Lower pretrain_learning_rate** if training is noisy:
   - Default 0.001; try 0.0005 for larger datasets

4. **Joint training** (`freeze_pretrained_cycle_nets=false`) may help if:
   - Auxiliary mapping is task-specific
   - You have enough pose supervision
   - Training is stable and not hitting local minima

5. **Monitor cycle loss components**:
   - If `IMU_cycle_loss >> Pressure_cycle_loss`, IMU prediction is hard
   - Consider `enable_imu_cycle_loss=false` to focus on pressure

## Checkpoint Format

Pretrained checkpoint saved by `pretrain_accelnet()`:
```python
{
    "model_state_dict": accel_net.state_dict(),
    "best_val_loss": 0.045678
}
```

Main training checkpoint (with cycle nets):
```python
{
    "epoch": 45,
    "model_state_dict": soleformer.state_dict(),
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "accel_net_state_dict": accel_net.state_dict(),
    "press_net_state_dict": press_net.state_dict(),
    "model_config": {...},
    ...
}
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Pretraining very slow | Large data, high batch size | Reduce batch_size or pretrain_epochs |
| Pretraining loss not decreasing | Bad initialization or task too hard | Increase pretrain_learning_rate or check data quality |
| Cycle loss becomes NaN | Frozen nets causing gradient issues | Set freeze_pretrained_cycle_nets=false |
| High MSE in pretraining but low cycle loss in main train | Nets overfit during pretrain | Early stopping or regularization in pretrain |

