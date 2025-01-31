import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
from mlp import ShallowMLP
import random
import os

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
# Load the features and labels from the generated JSON file
features_tensor, age_labels_tensor, sex_labels_tensor, cog_labels_tensor, secret1_tensor, secret2_tensor = ShallowMLP.load_json_data('generated_features.json')

# Create the dataloaders
train_loader, val_loader = ShallowMLP.create_dataloaders(features_tensor, age_labels_tensor, sex_labels_tensor, cog_labels_tensor, secret1_tensor, secret2_tensor)

# Instantiate the model
model = ShallowMLP(input_dim=128, hidden_dim=64)

# Train the model
model.train_model(train_loader=train_loader, val_loader=val_loader, num_epochs=100)

# Save the trained model
torch.save(model.state_dict(), "shallow_mlp.pth")
