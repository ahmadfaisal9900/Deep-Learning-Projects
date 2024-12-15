import os
import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load and prepare the test data as a graph
file_path = 'vision_beamforming_dataset_test.csv'
if os.path.exists(file_path):
    # Load and normalize data
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    df[['x_min', 'y_min', 'x_max', 'y_max']] = scaler.fit_transform(df[['x_min', 'y_min', 'x_max', 'y_max']])
    
    # Prepare features and labels
    features = torch.tensor(df[['x_min', 'y_min', 'x_max', 'y_max']].values, dtype=torch.float32).to(device)
    labels = torch.tensor(df['beam_index'].values, dtype=torch.long).to(device)
    
    # Calculate pairwise distances
    distances = torch.cdist(features, features, p=2)
    threshold = 0.5  # Define a distance threshold for connections

    # Create adjacency matrix based on the threshold
    adj_matrix = (distances < threshold).float()
    edge_index, _ = dense_to_sparse(adj_matrix)
    
    # Create a PyG Data object for the test dataset
    test_data = Data(x=features, edge_index=edge_index, y=labels)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
else:
    print("Test file not found.")
    exit()

# Define the GCN model architecture (same as the training model)
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

# Initialize the model, load the trained weights, and set it to evaluation mode
input_dim = 4
hidden_dim = 64
output_dim = 64  # Match the number of classes

model = OptimizedGCN(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("optimized_gcn_model.pth"))
model.eval()
print("Model loaded successfully.")

# Define the evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            
            # Calculate predictions and accuracy for each node
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total_samples += data.num_nodes  # Use the total number of nodes as sample count

    accuracy = correct / total_samples  # Divide by total number of nodes
    return total_loss / len(loader), accuracy

# Criterion for testing
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
