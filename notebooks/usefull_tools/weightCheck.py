import torch

weight_path = "./results/weight/best_skeleton_model.pth"  # Weight file path
checkpoint = torch.load(weight_path, map_location="cpu")  # Use CPU when GPU is unavailable


for name, param in checkpoint["model_state_dict"].items():
    print(name, param.shape)

for name, param in checkpoint["model_state_dict"].items():
    print(name, param.mean(), param.std())
