import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Use your specific model file

# Root directory where the images are stored
root_dir = Path(r'E:\Projects\Unfinished\Vision Aided Beamforming\Vision\Vision')  # Main folder

# Output directory where processed images will be saved
output_root_dir = Path(r'E:\Projects\Unfinished\Vision Aided Beamforming\Vision\Vision_BB')  # Output folder for images with bounding boxes

# Output directory for bounding box text files
bbox_output_dir = Path(r'E:\Projects\Unfinished\Vision Aided Beamforming\Vision\bounding_boxes')  # New directory for bounding box text files

# Get the class ID for 'car'
car_class_id = None
for cls_id, cls_name in model.names.items():
    if cls_name == 'car':
        car_class_id = cls_id
        break

if car_class_id is None:
    raise ValueError("Class 'car' not found in model.names")

# Walk through the directory structure
for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    dirpath = Path(dirpath)
    # Check if 'camera_data' is in the path
    if 'camera_data' in dirpath.parts:
        # Process image files in this directory
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = dirpath / filename

                # Perform object detection
                results = model(str(image_path))

                # Get detections
                detections = results[0].boxes  # This is a Boxes object

                # Filter detections for 'car' class
                car_detections = []
                for det in detections:
                    class_id = int(det.cls.cpu().numpy()[0])
                    if class_id == car_class_id:
                        car_detections.append(det)

                if not car_detections:
                    # No car detections, skip this image
                    continue

                # Select the detection with the highest confidence
                car_detections.sort(key=lambda det: det.conf.cpu().numpy(), reverse=True)
                best_det = car_detections[0]

                # Load image
                img = plt.imread(str(image_path))
                img_height, img_width = img.shape[:2]  # Get image dimensions

                fig, ax = plt.subplots()
                ax.imshow(img)

                # Get absolute bounding box coordinates
                x1, y1, x2, y2 = best_det.xyxy[0].cpu().numpy()
                
                conf = float(best_det.conf.cpu().numpy()[0])

                # Draw bounding box for the best detection (still in absolute pixel values for display)
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    edgecolor='r',
                    facecolor='none',
                    linewidth=2
                )
                ax.add_patch(rect)

                # Optional: Add class name and confidence score
                class_id = int(best_det.cls.cpu().numpy()[0])
                ax.text(
                    x1,
                    y1 - 10,
                    f"{model.names[class_id]}: {conf:.2f}",
                    color='red',
                    fontsize=12,
                    backgroundcolor='white'
                )

                bbox_coords_normalized = [x1, y1, x2, y2]

                ax.axis('off')

                # Save the figure without margins
                fig.tight_layout(pad=0)

                # Determine the output path for the image, maintaining the folder hierarchy
                relative_path = image_path.relative_to(root_dir)
                output_image_path = output_root_dir / relative_path
                output_dir = output_image_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save the image with bounding box
                fig.savefig(str(output_image_path), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                # Save normalized bounding box coordinates
                # Extract scenario name from the path
                scenario_name = None
                for part in dirpath.parts:
                    if part.lower().startswith('scenario'):
                        scenario_name = part
                        break
                if scenario_name is None:
                    # Handle case where scenario is not found
                    scenario_name = 'UnknownScenario'

                # Create output directory for bounding boxes
                bbox_scenario_dir = bbox_output_dir / scenario_name
                bbox_scenario_dir.mkdir(parents=True, exist_ok=True)

                # Save the normalized bounding box coordinates to a text file in the scenario subdirectory
                base_filename = image_path.stem
                bbox_txt_filename = f"{base_filename}_bbox.txt"
                bbox_txt_path = bbox_scenario_dir / bbox_txt_filename
                
                # Write normalized bounding box coordinates to the file
                with open(bbox_txt_path, 'w') as f:
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox_coords_normalized
                    f.write(f"{x1_norm},{y1_norm},{x2_norm},{y2_norm}\n")
