from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Clustering function to apply K-Means and add cluster labels
def add_cluster_labels(df):
    # Select the features to use for clustering
    feature_columns = ['x_min', 'y_min', 'x_max', 'y_max', 'aspect_ratio', 'box_center_x', 'box_center_y']
    feature_data = df[feature_columns].values
    unique_labels = df['beam_index'].nunique()
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=unique_labels, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_data)
    
    # Add cluster labels as a new column in the DataFrame
    df['cluster_label'] = cluster_labels
    return df, kmeans

# Plot the distribution of samples across clusters
def plot_label_distribution(df):
    cluster_counts = df['cluster_label'].value_counts()
    plt.figure(figsize=(10, 6))
    cluster_counts.sort_index().plot(kind='bar')
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Samples")
    plt.title("Number of Samples per Cluster")
    plt.show()

# Calculate clustering accuracy by comparing cluster labels with true labels
def calculate_clustering_accuracy(df):
    # Map each cluster to the most common true label within the cluster
    cluster_to_label = {}
    for cluster in df['cluster_label'].unique():
        mode_result = mode(df[df['cluster_label'] == cluster]['beam_index']).mode
        cluster_to_label[cluster] = mode_result
    
    # Map each cluster label to its assigned beam index
    df['predicted_label'] = df['cluster_label'].map(cluster_to_label)
    
    # Calculate accuracy
    accuracy = accuracy_score(df['beam_index'], df['predicted_label'])
    print(f"Clustering Accuracy: {accuracy * 100:.2f}%")

# Optionally visualize the clusters in 2D (PCA)
def plot_clusters(df, feature_columns):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df[feature_columns].values)
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['cluster_label'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Cluster Label')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustering Visualization (PCA Reduced)')
    plt.show()

# Paths to each dataset file
file_paths = {
    'train': 'vision_beamforming_dataset_train.csv',
    'test': 'vision_beamforming_dataset_test.csv',
    'val': 'vision_beamforming_dataset_val.csv'
}

# Step 1: Load datasets, normalize, and store them in TensorDatasets
data = {}

# Apply clustering and evaluation for each dataset split
for split, path in file_paths.items():
    if os.path.exists(path):
        # Load and preprocess data
        df = pd.read_csv(path)

        # Feature engineering
        df['width'] = df['x_max'] - df['x_min']
        df['height'] = df['y_max'] - df['y_min']
        df['aspect_ratio'] = df['width'] / (df['height'] + 1e-6)  # Avoid division by zero
        df['box_center_x'] = (df['x_min'] + df['x_max']) / 2
        df['box_center_y'] = (df['y_min'] + df['y_max']) / 2

        # Apply clustering and add cluster labels
        df, kmeans = add_cluster_labels(df)

        # Plot the distribution of samples across clusters
        plot_label_distribution(df)

        # Calculate and print the clustering accuracy
        calculate_clustering_accuracy(df)

        # Visualize clustering results in 2D (optional)
        feature_columns = ['x_min', 'y_min', 'x_max', 'y_max', 'aspect_ratio', 'box_center_x', 'box_center_y']
        plot_clusters(df, feature_columns)
        
        # Convert to PyTorch tensors if needed for training
        features = df[feature_columns + ['cluster_label']].values
        labels = df['beam_index'].values
        features = torch.tensor(features, dtype=torch.float32).cuda()
        labels = torch.tensor(labels, dtype=torch.long).cuda()

        # Store the dataset in a TensorDataset
        data[split] = TensorDataset(features, labels)
