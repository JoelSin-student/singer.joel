# Training processor
#
# TODO: Add "successfully" logs for each step.
# TODO: Allow model switching from the command line.
# TODO: Consider using quaternions.
# TODO: Re-check whether @staticmethod is necessary.
# TODO: Try random down/up-sampling as data augmentation.
# TODO: Try pretraining on large skeleton datasets (e.g., COCO) then fine-tuning.
# TODO: Evaluate first-derivative-only features.
# TODO: Explore collaboration with lightweight CNNs.
# TODO: Explore real-time processing.
# TODO: Test Mish activation function.
import pandas as pd
import numpy as np
import argparse
import joblib
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, calculate_grad, load_config,  PressureSkeletonDataset
from processor.util import print_config
from processor.model import Transformer_Encoder, Skeleton_Loss, train_Transformer_Encoder

def start(args):
    # Load YAML file
    config = load_config(args, args.config, args.model)

    # Set data path
    skeleton_dir = config["location"]["data_path"] + "/skeleton/"
    insole_dir   = config["location"]["data_path"] + "/Insole/"
    
    # Preprocess skeleton data and insole data
    # Load the data and combine the left and right insole data, then separate it into pressure data and IMU data.
    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)                           # Collect paired skeleton/insole file paths
    skeleton_df, insole_left_df, insole_right_df  = load_and_combine_data(skeleton_insole_datapath_pairs)   # Load skeleton, right-insole, and left-insole data
    pressure_lr_df, IMU_lr_df = restructure_insole_data(insole_left_df, insole_right_df)                    # Split pressure/IMU and merge left+right
    if config["train"]["use_gradient_data"] == True: calculate_grad()                                       # Add derivative features (experimental)
    # input_feature_np = np.concatenate([pressure_lr_df, IMU_lr_df], axis=1)

    # Temporary handling added for test4 data
    skeleton_df = skeleton_df.fillna(method='bfill').fillna(method='ffill')

    sigma=2
    pressure_lr_df = pressure_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))      # Temporarily added smoothing
    IMU_lr_df = IMU_lr_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))

    # Sprit data
    # Skeletal data, pressure data, and IMU data are each split 8:2
    train_pressure, val_pressure, train_IMU, val_IMU, train_skeleton, val_skeleton = train_test_split(
        pressure_lr_df, 
        IMU_lr_df, 
        skeleton_df,
        test_size=0.2,
        shuffle=False        # Keep off for sequence_len > 1 to preserve temporal order
    )

    # Initialize scaler
    pressure_normalizer = MinMaxScaler()
    imu_normalizer      = MinMaxScaler()
    # skeleton_scaler     = MinMaxScaler()

    # Fit the scaler on the training data and transform
    train_pressure_scaled = pressure_normalizer.fit_transform(train_pressure)  # Training pressure data (fit + transform)
    train_IMU_scaled      = imu_normalizer.fit_transform(train_IMU)            # Training IMU data (fit + transform)
    # train_skeleton_scaled = skeleton_scaler.fit_transform(train_skeleton)    # Training skeleton data (fit + transform)
    val_pressure_scaled   = pressure_normalizer.transform(val_pressure)        # Validation pressure data (transform)
    val_IMU_scaled        = imu_normalizer.transform(val_IMU)                  # Validation IMU data (transform)
    # val_skeleton_scaled   = skeleton_scaler.transform(val_skeleton)          # Validation skeleton data (transform)

    # save scaler
    # When I predict the model, I need to use same scaler.
    # joblib.dump(skeleton_scaler, './scaler/skeleton_scaler.pkl')

    # combine pressure data and IMU data
    train_input_feature = np.concatenate([train_pressure_scaled, train_IMU_scaled], axis=1)
    val_input_feature   = np.concatenate([val_pressure_scaled, val_IMU_scaled], axis=1) 

    # set final train parameters----------------------------------------------------------------------------
    parameters = {                                                      # TODO: consider shortening `parameters` variable name.
        # model
        "d_model"            : config["train"]["d_model"],
        "n_head"             : config["train"]["n_head"],
        "num_encoder_layer"  : config["train"]["num_encoder_layer"],    # TODO: standardize naming conventions (dim/num/len, etc.)
        "dropout"            : config["train"]["dropout"],

        # learning
        "num_epoch"          : config["train"]["epoch"],
        "batch_size"         : config["train"]["batch_size"],

        # optimize
        "learning_rate"      : config["train"]["learning_rate"],
        "weight_decay"       : config["train"]["weight_decay"],
        
        # loss function
        "loss_alpha"         : config["train"]["loss_alpha"],
        "loss_beta"          : config["train"]["loss_beta"],

        # others
        "use_gradient_data"  : config["train"]["use_gradient_data"],    # TODO: maybe rename to `use_grad`
        "sequence_len"       : config["train"]["sequence_len"],
        "input_dim"          : pressure_lr_df.shape[1] + IMU_lr_df.shape[1], # Total dims of pressure + gyro + acceleration
        "num_joints"         : skeleton_df.shape[1] // 3,                    # Divide by 3 for 3D coordinates
        "num_dims"           :  3
    }
    # <debug> print train parameters
    print_config(parameters)
    #-------------------------------------------------------------------------------------------------

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # make dataset
    train_dataset = PressureSkeletonDataset(train_input_feature, train_skeleton.to_numpy(), sequence_length=parameters["sequence_len"])   # train_skeleton_scaled
    val_dataset = PressureSkeletonDataset(val_input_feature, val_skeleton.to_numpy(), sequence_length=parameters["sequence_len"])         # val_skeleton_scaled
    
    # set dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=parameters["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # TODO: split ML model branches by mode
    # if(args.model == "transformer_encoder"):

    # initialize model
    model = Transformer_Encoder(
        input_dim          = parameters["input_dim"],
        d_model            = parameters["d_model"],
        nhead              = parameters["n_head"],
        num_encoder_layers = parameters["num_encoder_layer"],
        num_joints         = parameters["num_joints"],
        num_dims           = parameters["num_dims"],
        dropout            = parameters["dropout"]
    ).to(device)

    # set loss function
    criterion = Skeleton_Loss(alpha=parameters["loss_alpha"], beta=parameters["loss_beta"])

    # set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = parameters["learning_rate"],
        weight_decay = parameters["weight_decay"],
        betas        = (0.9, 0.999)
    )
    # set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'min',
        factor   = 0.5,
        patience = 5,
    )

    # start model training
    train_Transformer_Encoder(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer, 
        scheduler,
        num_epochs  = parameters["num_epoch"],
        save_path   = './weight/best_skeleton_model.pth',          # TODO: include date/model name in output filename
        device      = device
    )

    # keep checkpoint
    final_checkpoint = {
        'model_state_dict'      : model.state_dict(),
        'optimizer_state_dict'  : optimizer.state_dict(),
        'scheduler_state_dict'  : scheduler.state_dict(),
        'model_config': {
            'input_dim'         : parameters["input_dim"],
            'd_model'           : parameters["d_model"],
            'nhead'             : parameters["n_head"],
            'num_encoder_layers': parameters["num_encoder_layer"],
            'num_joints'        : parameters["num_joints"]
        }
    }
    torch.save(final_checkpoint, './weight/final_skeleton_model.pth')   # TODO: include date/model name in output filename
    return


@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Training Processor')

    # Basic settings
    parser.add_argument('--model', choices=['transformer_encoder','transformer', 'BERT'], default='transformer_encoder', help='Model selection')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML file')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--sequence_len', type=int, default=None)
    parser.add_argument('--use_gradient_data', type=str, default=None)

    # Model parameters
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--num_encoder_layer', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # Training parameters
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # Loss function parameters
    parser.add_argument('--loss_alpha', type=float, default=None)
    parser.add_argument('--loss_beta', type=float, default=0.1)

    # Processor
    # feeder
    # model

    return parser
