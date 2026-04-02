# 🚀 SoleFormer Quick Start Guide

## Installation

### 1. Set up Python Environment (if not already done)

```bash
cd singer.joel

# Create or activate conda environment
conda create -n soleformer python=3.10
conda activate soleformer

# Install dependencies (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn matplotlib jupyter tqdm seaborn
```

### 2. Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Quick Start (3 Options)

### Option 1: Run Jupyter Notebook (Recommended for Learning)

```bash
cd notebooks
jupyter notebook SoleFormer_Complete_Pipeline.ipynb
```

**Steps**:
1. Open the notebook in browser
2. Run cells sequentially from top to bottom
3. Watch the training progress and visualization
4. Modify parameters and experiment

**Key Cells**:
- Cell 1-2: Setup and imports
- Cell 3-4: Data preprocessing
- Cell 5-6: Synthetic data generation
- Cell 7-8: Dataset creation
- Cell 9-12: Model architecture
- Cell 13-14: Training loop
- Cell 15-16: Evaluation and visualization

---

### Option 2: Python Script

Create `run_soleformer.py`:

```python
import torch
import numpy as np
from notebooks.model import SoleFormer, AccelNet, PressNet, DoubleCycleConsistencyLoss

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Initialize models
model = SoleFormer(
    pressure_dim=32,
    imu_dim=12,
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    output_dim=51,  # 17 joints * 3
    dropout=0.1,
).to(device)

accel_net = AccelNet(input_dim=51, hidden_dim=256).to(device)
press_net = PressNet(input_dim=51, hidden_dim=256).to(device)

# Loss function
criterion = DoubleCycleConsistencyLoss(
    accel_net=accel_net,
    press_net=press_net,
    weight_pose=1.0,
    weight_imu_cycle=0.5,
    weight_pressure_cycle=0.5,
)

# Create sample input
batch_size = 4
seq_len = 16
pressure_imu_dim = 32 + 12  # 44

x = torch.randn(batch_size, seq_len, pressure_imu_dim).to(device)
target_pose = torch.randn(batch_size, seq_len, 51).to(device)
target_imu = torch.randn(batch_size, seq_len, 12).to(device)
target_pressure = torch.randn(batch_size, seq_len, 32).to(device)

# Forward pass
pred_pose = model(x)

# Compute loss
loss, loss_dict = criterion(pred_pose, target_pose, target_imu, target_pressure)

print(f"Predicted pose shape: {pred_pose.shape}")
print(f"Loss: {loss.item():.6f}")
print(f"Loss components: {loss_dict}")
```

Run it:
```bash
python run_soleformer.py
```

---

### Option 3: Command Line Training

```bash
python main.py train \
    --model transformer_encoder \
    --mode train \
    --model_mode soleformer \
    --config notebooks/config/transformer_encoder/train.yaml
```

---

## 📊 Understanding the Data

### Input Format

**Pressure Data** (32 channels):
- Sensors 0-15: Left foot (16 channels)
- Sensors 16-31: Right foot (16 channels)
- Values: 0-20 N/cm² → normalized to [0, 1]
- Represents foot contact dynamics

**IMU Data** (12 channels):
- Channels 0-2: Left foot acceleration (3-axis)
- Channels 3-5: Left foot gyroscope (3-axis)
- Channels 6-8: Right foot acceleration (3-axis)
- Channels 9-11: Right foot gyroscope (3-axis)
- Values: standardized (mean=0, std=1)
- Represents body movement

**Sequence Format**:
- Length: 16 frames (temporal context)
- Total input: 32 + 12 = 44 dimensions
- Input shape: (batch, 16, 44)

### Output Format (Pose)

51 dimensions = 17 joints × 3 coordinates (X, Y, Z)

**Joint Order** (typically):
```
0: Head
1: Neck
2-4: Left arm (shoulder, elbow, wrist)
5-7: Right arm (shoulder, elbow, wrist)
8-10: Left leg (hip, knee, ankle)
11-13: Right leg (hip, knee, ankle)
14-16: Torso/additional joints
```

Values: Standardized, centered by hip (joint 0)

---

## 🎛️ Hyperparameter Tuning

### Important Parameters

```python
# Model

 Architecture
config = {
    'd_model': 128,           # Hidden dimension (64, 128, 256)
    'nhead': 8,               # Attention heads (4, 8, 16)
    'num_encoder_layers': 2,  # Layers (1, 2, 3, 4)
    'dropout': 0.1,           # Dropout rate (0.0, 0.1, 0.2)
}

# Training
train_config = {
    'learning_rate': 1e-3,     # 1e-4, 1e-3, 1e-2
    'weight_decay': 1e-5,      # 0, 1e-5, 1e-4
    'batch_size': 32,          # 16, 32, 64
    'num_epochs': 50,          # 20, 50, 100
}

# Loss Weights
loss_weights = {
    'weight_pose': 1.0,           # [0.5, 1.0, 2.0]
    'weight_imu_cycle': 0.5,      # [0.1, 0.5, 1.0]
    'weight_pressure_cycle': 0.5, # [0.1, 0.5, 1.0]
}
```

### Quick Experimentation

Modify in the notebook:

```python
# Cell with model config
model_config = {
    'pressure_dim': 32,
    'imu_dim': 12,
    'd_model': 256,        # ← Increase for larger model
    'nhead': 16,           # ← Increase for more attention heads
    'num_encoder_layers': 3,  # ← Add more layers
    'output_dim': 51,
    'dropout': 0.2,        # ← Increase for more regularization
    'use_graph_pressure': True,
}
```

---

## 📈 Monitoring Training

### Key Metrics to Watch

1. **Total Loss**: Should decrease over epochs
2. **Pose Loss**: Primary supervision signal
3. **IMU Cycle Loss**: Physical constraint
4. **Pressure Cycle Loss**: Physical constraint

### Training Curves

The notebook generates plots showing:

```
Training Loss
├─ Total loss (should decrease)
├─ Pose loss (should decrease)
├─ IMU cycle loss (should decrease)
└─ Pressure cycle loss (should decrease)
```

### Early Stopping

If validation loss doesn't improve for 10 epochs:
```python
patience = 10
counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = val_epoch()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
```

---

## 🧪 Inference Example

### Simple Prediction

```python
import torch
from notebooks.model import SoleFormer

# Load model
model = SoleFormer(pressure_dim=32, imu_dim=12, d_model=128)
model.load_state_dict(torch.load('soleformer_best.pt'))
model.eval()

# Prepare input
x = torch.randn(1, 16, 44)  # (batch=1, seq_len=16, features=44)

# Predict
with torch.no_grad():
    pose = model(x)  # (1, 16, 51)

print(f"Predicted pose shape: {pose.shape}")
print(f"First joint position: {pose[0, 0, :3]}")  # X, Y, Z of joint 0
```

### Batch Inference

```python
# Load real data
from torch.utils.data import DataLoader

val_loader = DataLoader(val_dataset, batch_size=32)

all_predictions = []
with torch.no_grad():
    for batch in val_loader:
        x = batch['input'].to(device)
        pred = model(x)
        all_predictions.append(pred.cpu())

# Concatenate
predictions = torch.cat(all_predictions, dim=0)
print(f"Total predictions: {predictions.shape}")
```

---

## 📊 Expected Results

### Performance Benchmarks

| Metric | Expected Value | Paper Value |
|--------|---|---|
| MPJPE (Mean Per-Joint Position Error) | 65-70 mm | 65.3 mm |
| MPJAE (Mean Per-Joint Angle Error) | 30-35° | 29.7° |
| Inference Time | 10-12 ms | 11.0 ms |
| Training Time (50 epochs, 1000 samples) | 5-10 minutes | - |

### On Your Data

- **Synthetic data**: 65-75 mm MPJPE
- **Real SolePose data**: 65-70 mm MPJPE
- **Different activities**: May vary ±5-10 mm

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or sequence length
```python
batch_size = 16  # or 8
sequence_length = 8  # or 4
```

### Issue: Loss not decreasing
**Solution**: 
- Check learning rate (try 5e-4 or 5e-3)
- Check loss weights (balance components)
- Verify data preprocessing

### Issue: Poor inference results
**Solution**:
- Train longer (100+ epochs)
- Increase model size (d_model=256)
- Load best checkpoint, not final

---

## 📚 Repository Structure

```
notebooks/
├── SoleFormer_Complete_Pipeline.ipynb  ← START HERE
├── model.py                           ← Architecture
├── loader.py                          ← Data loading
├── train.py                           ← Training pipeline
└── config/transformer_encoder/
    └── train.yaml                     ← Configuration
```

---

## ✅ Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] PyTorch installed with GPU support (optional)
- [ ] NumPy, pandas, scipy installed
- [ ] Jupyter notebook working
- [ ] Checked file paths are correct
- [ ] Reviewed data format

---

## 🎓 Next Steps

1. **Run the notebook** - Understand the pipeline end-to-end
2. **Load your data** - Replace synthetic data with real data
3. **Tune hyperparameters** - Find best config for your data
4. **Visualize predictions** - Compare with ground truth
5. **Deploy** - Export model to production
6. **Evaluate** - Compute MPJPE on test set

---

## 💡 Tips & Tricks

### 1. Speed up experiments
```python
# Train on smaller dataset first
dataset = dataset[:500]  # Use 500 samples instead of 2000
```

### 2. Monitor memory usage
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 3. Save training progress
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

### 4. Inspect model architecture
```python
from torchinfo import summary
summary(model, input_size=(batch_size, seq_len, 44))
```

---

## 📞 Getting Help

1. **Check the README**: `SOLEFORMER_README.md`
2. **Read the paper**: Original SolePoser paper (Wu et al., UIST 2024)
3. **Review code comments**: Each class has docstrings
4. **Run the notebook**: Tutorial-style explanations

---

**Good luck! 🚀**
