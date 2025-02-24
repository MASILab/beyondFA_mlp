import json
import torch
import numpy as np
from mlp import ShallowMLP

# Function to load the features from the JSON file
def load_json_data(json_file):
    # Load the .json file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert the list directly to a NumPy array
    features = np.array(data)
    
    # Convert the NumPy array to a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    return features_tensor

# Function for running inference
def run_inference(json_file):
    input_features = load_json_data(json_file)
    model = ShallowMLP(input_dim=128, hidden_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure the model is loaded with pre-trained weights
    model.load_state_dict(torch.load("shallow_mlp.pth"))
    
    # Explicitly set model to eval mode
    model.train(False)  # Ensure explicitly in eval mode
    model.eval()        # Reconfirm the mode



    # Run inference without backpropagation
    with torch.no_grad():
        input_features = input_features.to(device)
        age_pred, sex_pred, cog_pred, secret1_pred, secret2_pred = model(input_features)

        # Output the predictions
        print(f"Predicted Age: {age_pred.item():.0f}")
        print(f"Predicted Sex: {torch.sigmoid(sex_pred).item():.0f}")
        print(f"Predicted Cognitive Status: {torch.argmax(cog_pred).item()}")
        print(f"Predicted Secret Variable 1: {secret1_pred.item():.2f} (A reasonable value expectation is -20 to 38; you may see values outside this range)")
        print(f"Predicted Secret Variable 2: {secret2_pred.item():.2f} (A reasonable value expectation is 0 to 100; you should not see values outside this range)")


json_file='test_features.json' # Replace with the path to your JSON file; should be named for your subject name
run_inference(json_file)


