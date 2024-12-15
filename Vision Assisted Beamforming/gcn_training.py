import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
import torch
print(torch.__version__)
# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Paths to each dataset file
file_paths = {
    'train': 'vision_beamforming_dataset_train.csv',
    'test': 'vision_beamforming_dataset_test.csv',
    'val': 'vision_beamforming_dataset_val.csv'
}

# Step 1: Load datasets, normalize, and prepare sparse graph data
scaler = StandardScaler()
datasets = {}

for split, path in file_paths.items():
    if os.path.exists(path):
        # Load and normalize data
        df = pd.read_csv(path)
        df[['x_min', 'y_min', 'x_max', 'y_max']] = scaler.fit_transform(df[['x_min', 'y_min', 'x_max', 'y_max']])
        
        features = torch.tensor(df[['x_min', 'y_min', 'x_max', 'y_max']].values, dtype=torch.float32).to(device)
        labels = torch.tensor(df['beam_index'].values, dtype=torch.long).to(device)
        
        from torch_geometric.utils import dense_to_sparse

        # Calculate pairwise distances
        distances = torch.cdist(features, features, p=2)
        threshold = 0.5  # Define a distance threshold for connections

        # Create adjacency matrix based on threshold
        adj_matrix = (distances < threshold).float()
        edge_index, _ = dense_to_sparse(adj_matrix)
        data = Data(x=features, edge_index=edge_index, y=labels)
        datasets[split] = data

class OptimizedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(OptimizedGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        
        # Fully connected layer for each node
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GCN layers with LayerNorm and ReLU
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.relu(x)
        
        # Fully connected layers for node-level classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Each node has its own output
        
        return x


# Step 3: Training Function
def train(model, optimizer, loader, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Step 4: Evaluation Function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# Step 5: Initialize Model, Optimizer, and DataLoader
input_dim = 4  # Number of features
hidden_dim = 64
output_dim = 64 # Number of classes

model = OptimizedGCN(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Load individual samples for batching
train_loader = DataLoader([datasets['train']], batch_size=32, shuffle=True)  # Reduced batch size
val_loader = DataLoader([datasets['val']], batch_size=32, shuffle=False)

# Step 6: Train the Model
num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, optimizer, train_loader, criterion)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the Model
torch.save(model.state_dict(), "optimized_gcn_model.pth")
