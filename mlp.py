import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np

class ShallowMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dims=(1, 1, 3, 1, 1), learning_rate=1e-3):
        super(ShallowMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2_age = nn.Linear(hidden_dim, output_dims[0])  # Regression: Age (continuous)
        self.fc2_sex = nn.Linear(hidden_dim, output_dims[1])  # Binary Classification: Sex
        self.fc2_cog = nn.Linear(hidden_dim, output_dims[2])  # Multiclass: Cognitive status
        self.fc2_secret1 = nn.Linear(hidden_dim, output_dims[3]) # Regression: Secret variable 1
        self.fc2_secret2 = nn.Linear(hidden_dim, output_dims[4])  # Regression: Secret variable 2
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        age_pred = self.fc2_age(x)
        sex_pred = self.fc2_sex(x)
        cog_pred = self.fc2_cog(x)
        secret1_pred = self.fc2_secret1(x)
        secret2_pred = self.fc2_secret2(x)
        return age_pred, sex_pred, cog_pred, secret1_pred, secret2_pred

    def train_model(self, train_loader, val_loader, num_epochs=100):
        criterion_age = nn.MSELoss()
        criterion_sex = nn.BCEWithLogitsLoss()
        criterion_cog = nn.CrossEntropyLoss()
        criterion_secret1 = nn.MSELoss()
        criterion_secret2 = nn.MSELoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                features, age_labels, sex_labels, cog_labels, secret1_labels, secret2_labels = [b.to(device) for b in batch]

                self.optimizer.zero_grad()
                age_pred, sex_pred, cog_pred, secret1_pred, secret2_pred = self(features)

                loss_age = criterion_age(age_pred.squeeze(), age_labels.squeeze())
                loss_sex = criterion_sex(sex_pred.squeeze(), sex_labels.squeeze())
                loss_cog = criterion_cog(cog_pred, cog_labels)
                loss_secret1 = criterion_secret1(secret1_pred, secret1_labels)
                loss_secret2 = criterion_secret2(secret2_pred, secret2_labels)

                loss = loss_age + loss_sex + loss_cog + loss_secret1 + loss_secret2
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Validation phase
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    features, age_labels, sex_labels, cog_labels, secret1_labels, secret2_labels = [b.to(device) for b in batch]
                    age_pred, sex_pred, cog_pred, secret1_pred, secret2_pred = self(features)

                    loss_age = criterion_age(age_pred.squeeze(), age_labels.squeeze())
                    loss_sex = criterion_sex(sex_pred.squeeze(), sex_labels.squeeze())
                    loss_cog = criterion_cog(cog_pred, cog_labels)
                    loss_secret1 = criterion_secret1(secret1_pred, secret1_labels)
                    loss_secret2 = criterion_secret2(secret2_pred, secret2_labels)
                    val_loss += (loss_age + loss_sex + loss_cog + loss_secret1 + loss_secret2).item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def inference(self, input_data):
        self.eval()
        with torch.no_grad():
            output = self(input_data)
        return output

    @staticmethod
    def load_json_data(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        features = np.array([item['features'] for item in data])
        age_labels = np.array([item['age'] for item in data])
        sex_labels = np.array([item['sex'] for item in data])
        cog_labels = np.array([item['cognitive_status'] for item in data])
        secret1_labels = np.array([item['secret 1'] for item in data])
        secret2_labels = np.array([item['secret 2'] for item in data])

        # Convert them to PyTorch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32)
        sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.float32)
        cog_labels_tensor = torch.tensor(cog_labels, dtype=torch.long)
        secret1_labels_tensor = torch.tensor(secret1_labels, dtype=torch.float32)
        secret2_labels_tensor = torch.tensor(secret2_labels, dtype=torch.float32)

        return features_tensor, age_labels_tensor, sex_labels_tensor, cog_labels_tensor, secret1_labels_tensor, secret2_labels_tensor

    @staticmethod
    def create_dataloaders(features_tensor, age_labels_tensor, sex_labels_tensor, cog_labels_tensor, secret1_tensor, secret2_tensor, batch_size=32):
        dataset = TensorDataset(features_tensor, age_labels_tensor, sex_labels_tensor, cog_labels_tensor, secret1_tensor, secret2_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader



