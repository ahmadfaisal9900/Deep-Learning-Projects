import pandas as pd
import os
from typing import List, Dict, Union

def load_bounding_box(bbox_path: str) -> List[float]:
    """Loads bounding box coordinates from a text file and returns them as a list of floats."""
    with open(bbox_path, 'r') as file:
        # Read the line and split it by commas to get each coordinate
        coordinates = file.readline().strip().split(',')
        return list(map(float, coordinates))  # Convert each coordinate to float


def normalize_bounding_box(coordinates: List[float], img_width: int, img_height: int) -> List[float]:
    """Normalizes bounding box coordinates relative to image dimensions."""
    x_min, y_min, x_max, y_max = coordinates
    return [x_min / img_width, y_min / img_height, x_max / img_width, y_max / img_height]

def create_dataset(image_base_path: str, bbox_base_path: str, scenarios: List[int], img_width: int, img_height: int) -> Dict[str, List[Dict[str, Union[List[float], int]]]]:
    """
    Creates a dataset with bounding boxes and labels for each scenario and data split.

    Parameters:
    - image_base_path: str - Base directory where the image scenario folders are located.
    - bbox_base_path: str - Base directory where the bounding box scenario folders are located.
    - scenarios: List[int] - List of scenario numbers to process.
    - img_width: int - Width of the images, used for bounding box normalization.
    - img_height: int - Height of the images, used for bounding box normalization.

    Returns:
    - Dict[str, List[Dict[str, Union[List[float], int]]]] - Dataset with train, test, and validation splits.
    """
    dataset = {'train': [], 'test': [], 'validation': []}
    
    for scenario in scenarios:
        for split in ['train', 'test', 'validation']:
            # Adjust the CSV filename pattern
            csv_filename = f"scenario{scenario}_dev_{split}.csv"
            csv_path = os.path.join(image_base_path, f"Scenario{scenario}", "development_dataset", csv_filename)
            
            # Check if the CSV file exists
            if not os.path.isfile(csv_path):
                print(f"CSV file not found at path: {csv_path}")
                continue  # Skip if the file is not found
            
            data = pd.read_csv(csv_path)
            
            for _, row in data.iterrows():
                # Extract information from the CSV
                img_path = row['unit1_rgb_1']
                beam_index = int(row['beam_index_1'])
                
                # Construct the path to the bounding box file
                img_filename = os.path.basename(img_path)  # Get the image filename
                bbox_filename = img_filename.replace('.jpg', '_bbox.txt')
                bbox_path = os.path.join(bbox_base_path, f"Scenario{scenario}", bbox_filename)
                
                # Load and normalize bounding box coordinates
                bbox_coordinates = load_bounding_box(bbox_path)
                normalized_bbox = normalize_bounding_box(bbox_coordinates, img_width, img_height)
                
                # Append data to the dataset dictionary
                dataset[split].append({
                    'bounding_box': normalized_bbox,
                    'beam_index': beam_index
                })
                
    return dataset

# Parameters
image_base_path: str = "E:\\Projects\\Unfinished\\Vision Aided Beamforming\\Vision\\Vision_BB\\"  # Base path for the images
bbox_base_path: str = "E:\\Projects\\Unfinished\\Vision Aided Beamforming\\Vision\\bounding_boxes\\"  # Base path for the bounding boxes
scenarios: List[int] = list(range(5, 10))  # Assuming scenarios from Scenario5 to Scenario9
img_width: int = 100  # Replace with your actual image width
img_height: int = 100  # Replace with your actual image height

# Create the dataset
dataset: Dict[str, List[Dict[str, Union[List[float], int]]]] = create_dataset(image_base_path, bbox_base_path, scenarios, img_width, img_height)

# Print sample from the dataset
for split, data in dataset.items():
    print(f"{split.capitalize()} data sample:", data[:3])  # Display a few samples for each split
