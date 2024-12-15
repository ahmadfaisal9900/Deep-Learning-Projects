import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
# Load test data

file_path = 'vision_beamforming_dataset_test.csv'
if os.path.exists(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Calculate variance for analysis
    variances = df[['x_min', 'y_min', 'x_max', 'y_max', 'beam_index']].var().to_dict()
    
    # Feature engineering: calculate aspect ratio, box area, and box center
    df['width'] = df['x_max'] - df['x_min']
    df['height'] = df['y_max'] - df['y_min']
    df['aspect_ratio'] = df['width'] / (df['height'] + 1e-6)  # Adding small value to avoid division by zero
    df['box_center_x'] = (df['x_min'] + df['x_max']) / 2
    df['box_center_y'] = (df['y_min'] + df['y_max']) / 2
    
    # Define the final features to include in the model
    feature_columns = ['x_min', 'y_min', 'x_max', 'y_max', 'aspect_ratio', 'box_center_x', 'box_center_y']
    features = df[feature_columns].values
    labels = df['beam_index'].values
    
    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32).cuda()
    labels = torch.tensor(labels, dtype=torch.long).cuda()  # Assuming beam_index is categorical
    
    test_data = TensorDataset(features, labels)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
# Step 2: Define the FCNN Model with increased depth, dropout, and batch normalization
class FCNN(nn.Module):
    def __init__(self, num_classes):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # Adapted to 6x1 input shape
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Updated fully connected layer dimensions to match new conv output
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 1, 64),  # Adjusted based on the new conv output
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    
    def forward(self, x):
        x = x.view(-1, 1, 7, 1)  # Reshape input to 1x6x1 for CNN
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load model weights
num_classes = 64
fcnn_model = FCNN(num_classes).cuda()
fcnn_model.load_state_dict(torch.load('fcnn_model_optimized.pth'))

cnn_model = CNN(num_classes).cuda()
cnn_model.load_state_dict(torch.load('cnn_model_optimized.pth'))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Function to test a model
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            # For CNN, reshape input to fit conv layers
            if isinstance(model, CNN):
                features = features.view(-1, 1, 7, 1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Test FCNN Model
fcnn_test_loss, fcnn_accuracy = test_model(fcnn_model, test_loader)
print(f"FCNN Test Loss: {fcnn_test_loss:.4f}, FCNN Test Accuracy: {fcnn_accuracy:.2f}%")

# Test CNN Model
cnn_test_loss, cnn_accuracy = test_model(cnn_model, test_loader)
print(f"CNN Test Loss: {cnn_test_loss:.4f}, CNN Test Accuracy: {cnn_accuracy:.2f}%")
