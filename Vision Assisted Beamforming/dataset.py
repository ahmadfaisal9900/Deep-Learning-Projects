import os
import pandas as pd

# Directory paths
bounding_boxes_dir = 'Vision/bounding_boxes'
vision_dir = 'Vision/Vision'

# Function to create a dataset for a specific file type (train, test, or val)
def create_dataset(file_type):
    dataset = []
    for scenario in range(5, 10):
        scenario_folder = os.path.join(bounding_boxes_dir, f'Scenario{scenario}')
        
        # Load bounding boxes for this scenario
        bounding_boxes = {}
        for txt_file in os.listdir(scenario_folder):
            with open(os.path.join(scenario_folder, txt_file), 'r') as f:
                coords = f.read().strip().split(',')
                bounding_boxes[txt_file.replace('_bbox.txt', '')] = list(map(float, coords))

        # Load vision data for each file type
        dataset_folder = os.path.join(vision_dir, f'Scenario{scenario}', 'development_dataset')
        csv_file = f'scenario{scenario}_dev_{file_type}.csv'
        csv_path = os.path.join(dataset_folder, csv_file)
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                image_key = row['unit1_rgb_1'].split('/')[-1].replace('.jpg', '')
                beam_index = row['beam_index_1']
                if image_key in bounding_boxes:
                    bbox_coords = bounding_boxes[image_key]
                    dataset.append([*bbox_coords, beam_index])

    # Convert to DataFrame and save
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'beam_index']
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(f'vision_beamforming_dataset_{file_type}.csv', index=False)

# Create separate datasets for train, test, and val
for file_type in ['train', 'test', 'val']:
    create_dataset(file_type)
