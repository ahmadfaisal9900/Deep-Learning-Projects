import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Check for CUDA
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can run on the GPU.")
else:
    print("CUDA is not available. PyTorch will run on the CPU.")

# Paths to each dataset file
file_paths = {
    'train': 'vision_beamforming_dataset_train.csv',
    'val': 'vision_beamforming_dataset_val.csv'
}

# Function to plot class distribution
def plot_class_distribution(df, split_name):
    class_counts = df['beam_index'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar')
    plt.xlabel("Class (Beam Index)")
    plt.ylabel("Number of Samples")
    plt.title(f"Class Distribution after Balancing - {split_name}")
    plt.show()

# Step 1: Load datasets, balance classes, normalize, and store them in TensorDatasets
variances = {}
data = {}

for split, path in file_paths.items():
    if os.path.exists(path):
        # Load data
        df = pd.read_csv(path)
        
        # Feature engineering: calculate aspect ratio and box center
        df['width'] = df['x_max'] - df['x_min']
        df['height'] = df['y_max'] - df['y_min']
        df['aspect_ratio'] = df['width'] / (df['height'] + 1e-6)  # Avoid division by zero
        df['box_center_x'] = (df['x_min'] + df['x_max']) / 2
        df['box_center_y'] = (df['y_min'] + df['y_max']) / 2
        
        # Define the features to be used in balancing
        feature_columns = ['x_min', 'y_min', 'x_max', 'y_max', 'aspect_ratio', 'box_center_x', 'box_center_y']
        
        # Balance the classes by adding noise to minority classes
        max_count = df['beam_index'].value_counts().max()
        balanced_df = []

        for label in df['beam_index'].unique():
            label_df = df[df['beam_index'] == label]
            if len(label_df) < max_count:
                # Duplicate with small noise to reach max_count
                noise = np.random.normal(0, 0.01, size=(max_count - len(label_df), len(feature_columns)))
                new_samples = label_df.sample(n=max_count - len(label_df), replace=True).copy()
                new_samples[feature_columns] += noise
                label_df = pd.concat([label_df, new_samples], ignore_index=True)
            balanced_df.append(label_df)

        # Combine all balanced data for this split
        df = pd.concat(balanced_df, ignore_index=True)
        
        # Plot class distribution after balancing
        # plot_class_distribution(df, split)
        
        # Calculate variance for analysis
        variances[split] = df[['x_min', 'y_min', 'x_max', 'y_max', 'beam_index']].var().to_dict()
        
        # Prepare features and labels for the model
        features = df[feature_columns].values
        labels = df['beam_index'].values
        
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32).cuda()
        labels = torch.tensor(labels, dtype=torch.long).cuda()  # Assuming beam_index is categorical
        
        # Store the dataset in a TensorDataset
        data[split] = TensorDataset(features, labels)

# Display variances
print("Variance across train and val splits:", variances)

# Step 2: Define the FCNN Model with increased depth, dropout, and batch normalization
class FCNN(nn.Module):
    def __init__(self, num_classes):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.apply(self.init_weights)
    
    def forward(self, x):
        return self.fc(x)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier Initialization
            m.bias.data.fill_(0.01)

# Step 3: Define the CNN Model with increased depth, dropout, and batch normalization

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
        
        self.apply(self.init_weights)
    
    def forward(self, x):
        x = x.view(-1, 1, 7, 1)  # Reshape input to 1x6x1 for CNN
        x = self.conv(x)
        x = self.fc(x)
        return x
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier Initialization
            m.bias.data.fill_(0.01)

# Step 4: Training Function with Gradient Clipping and Learning Rate Scheduler
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1000):
    model = model.cuda()
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            running_train_loss += loss.item()
        
        # Compute average training loss
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
    
    return train_losses, val_losses

# Step 5: Set Up Data Loaders
train_loader = DataLoader(data['train'], batch_size=128, shuffle=True)  # Increased batch size
val_loader = DataLoader(data['val'], batch_size=128, shuffle=False)

# Determine number of classes based on unique labels in the training data
num_classes = 64

# Step 6: Initialize, Train, and Save FCNN Model with Weight Decay in Optimizer and Learning Rate Scheduler
fcnn = FCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(fcnn.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

print("Training FCNN...")
fcnn_train_losses, fcnn_val_losses = train_model(fcnn, train_loader, val_loader, criterion, optimizer, scheduler)

# Save the FCNN model
torch.save(fcnn.state_dict(), 'fcnn_model_optimized.pth')

# Step 7: Initialize, Train, and Save CNN Model
cnn = CNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(cnn.parameters(), lr=0.005, weight_decay=1e-5)  # Added weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

print("Training CNN...")
cnn_train_losses, cnn_val_losses = train_model(cnn, train_loader, val_loader, criterion, optimizer, scheduler)

# Save the CNN model
torch.save(cnn.state_dict(), 'cnn_model_optimized.pth')

# Step 8: Plot Training and Validation Losses
plt.figure(figsize=(12, 6))
plt.plot(fcnn_train_losses, label='FCNN Training Loss')
plt.plot(fcnn_val_losses, label='FCNN Validation Loss')
plt.plot(cnn_train_losses, label='CNN Training Loss')
plt.plot(cnn_val_losses, label='CNN Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves with Optimizations')
plt.legend()
plt.show()
