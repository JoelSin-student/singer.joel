# File for building deep learning models
#
#
# 
#
import pandas as pd 
import math
import time
import datetime
import os
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1):
        super().__init__()

        self.num_joints = num_joints
        self.num_dims = num_dims

        # first layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model),  # Adjust dimensionality
            nn.LayerNorm(d_model),          # Stabilize training
            nn.ReLU(),                      # Activation function (Mish is another option)
            nn.Dropout(dropout),            # Prevent overfitting
            nn.Linear(d_model, d_model))    # Adjust dimensionality

        # positional encoding
        self.positional_encoder = PositionalEncoding(d_model)

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,              # Model dimension (larger can increase representation power)
                nhead=nhead,                  # Number of attention heads
                dim_feedforward=d_model * 4,  # Feedforward hidden size (may need tuning)
                dropout=dropout,              # Dropout regularization
                batch_first=True,             # 
                norm_first=False),            # 7/24: changed from True to False
            num_layers=num_encoder_layers
        )
        
        # output layer
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),                # Adjust dimensionality
            nn.LayerNorm(d_model),                      # Stabilize training
            nn.ReLU(),                                  # Activation function (Mish is another option)
            nn.Dropout(dropout),                        # Dropout regularization
            nn.Linear(d_model, d_model),                # Adjust dimensionality
            nn.ReLU(),                                  # Activation function
            nn.Linear(d_model, num_joints * num_dims))  # Final output layer (joint coordinates)
        
        # scaling factor
        self.output_scale = nn.Parameter(torch.ones(1)) 
    
    def forward(self, x):
        features = self.feature_extractor(x)                     # Feature extraction
        features = self.positional_encoder(features)             # positional encoding
        transformer_output = self.transformer_encoder(features)  # Transformer encoder pass
        last_time_step_output = transformer_output[:, -1, :]     # Use representation of last time step
        output = self.output_decoder(last_time_step_output)      # Generate output
        output = output * self.output_scale                      # Output scaling
        return output

class Skeleton_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)                  # Reconstruction loss
        pred_var = pred.var(dim=0).mean()                    # Mean per-joint variance across the batch
        return self.alpha * mse_loss - self.beta * pred_var  # Subtract variance to penalize collapsed predictions
    

def train_Transformer_Encoder(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    save_path,
    device,
    min_output_variance=1e-5,
    collapse_penalty_weight=1000.0,
):
    best_val_loss = float('inf')
    best_score = float('inf')

    model_config = {
        'input_dim': model.feature_extractor[0].in_features,
        'd_model': model.feature_extractor[0].out_features,
        'nhead': model.transformer_encoder.layers[0].self_attn.num_heads,
        'num_encoder_layers': len(model.transformer_encoder.layers),
        'num_joints': model.num_joints,
    }

    # Record start time
    start_time = time.time()
    pre_time   = start_time
    now_time   = datetime.datetime.now()
    print(f"\n[train started at {now_time.strftime("%H:%M")}]")
    
    # Training phase
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for pressure, skeleton in train_pbar:
            pressure = pressure.to(device)     # Move data to device
            skeleton = skeleton.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pressure)
            loss = criterion(outputs, skeleton)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_output_var = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
        with torch.no_grad():
            for pressure, skeleton in val_pbar:
                pressure = pressure.to(device)  # Move data to device
                skeleton = skeleton.to(device)
                
                outputs = model(pressure)
                loss = criterion(outputs, skeleton)
                val_loss += loss.item()
                val_output_var += float(outputs.var(dim=0, unbiased=False).mean().item())
        
        # Compute average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_output_var = val_output_var / len(val_loader)

        collapse_penalty = max(0.0, float(min_output_variance) - avg_val_output_var)
        score = avg_val_loss + float(collapse_penalty_weight) * collapse_penalty
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Compute elapsed time
        end_time = time.time()
        epoch_time_total_s   = end_time - pre_time
        elaps_time_total_s   = end_time - start_time
        epoch_time_m         = int(epoch_time_total_s//60)
        epoch_time_s         = int(epoch_time_total_s%60)
        elaps_time_m         = int(elaps_time_total_s//60)
        elaps_time_s         = int(elaps_time_total_s%60)

        # Estimate finish time
        remaining_epochs = num_epochs - (epoch + 1)
        est_remaining_time_s = epoch_time_total_s * remaining_epochs
        est_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_time_s)
        est_finish_str  = est_finish_time.strftime("%H:%M")

        print(f'------------ Epoch {epoch+1}/{num_epochs} ------------\n'
              f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n'
              f'Val Output Var: {avg_val_output_var:.8f} | Score: {score:.6f}\n'
              f'LR        : {current_lr:.5f}\n'
              f'Time/epoch: {epoch_time_m}m {epoch_time_s}s | Total: {elaps_time_m}m {elaps_time_s}s\n'
              f'Estimated Finish: {est_finish_str}')
        
        pre_time = end_time
        
        # Save best model checkpoint using loss + collapse penalty score.
        if score < best_score:
            best_val_loss = avg_val_loss
            best_score = score
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_score': best_score,
                'best_val_output_variance': avg_val_output_var,
                'min_output_variance': float(min_output_variance),
                'collapse_penalty_weight': float(collapse_penalty_weight),
                'model_config': model_config,
            }
            torch.save(checkpoint, save_path)
            print(f'>> Model saved at epoch {epoch+1} (score={best_score:.6f})')


def load_Transformer_Encoder(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, scheduler, epoch, best_val_loss


def save_predictions(predictions, model, frame_indices=None, output_stem=None):
    # Convert predictions to DataFrame
    num_joints = predictions.shape[1] // 3
    columns = []
    for i in range(num_joints):
        columns.extend([f'X.{i}', f'Y.{i}', f'Z.{i}'])

    output_name = f'Predicted_skeleton_{output_stem}' if output_stem else 'Predicted_skeleton'
    output_file = f'./results/output/{output_name}.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_predictions = pd.DataFrame(predictions, columns=columns)
    if frame_indices is not None:
        if len(frame_indices) != len(df_predictions):
            raise ValueError(
                f"Length mismatch: frame_indices has {len(frame_indices)} rows but "
                f"predictions has {len(df_predictions)} rows"
            )
        df_predictions.insert(0, 'Frame', frame_indices)
    df_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# Model inference
def predict_Transformer_Encoder(model, pressure_data):
    model.eval()
    with torch.no_grad():
        pressure_tensor = torch.FloatTensor(pressure_data)
        predictions = model(pressure_tensor)
    return predictions.numpy()