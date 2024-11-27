import cv2
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
import time
from collections import Counter
def get_sign_language_label(class_index):
    # Assuming you have a list of labels in the same order as the class indices
    labels = ['bad', 'best', 'glad', 'sad', 'scared', 'stiff', 'surprise']
    return

# Load the trained ViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("E:\\Projects\\Sign Language\\PkSLMNM_Model").to(device)

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("E:\\Projects\\Sign Language\\PkSLMNM_Model")

# Define the video capture
cap = cv2.VideoCapture(0)  # You may change the parameter to the appropriate device index or video file

# Set up real-time inference
accumulated_frames = []
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    inputs = feature_extractor(images=frame, return_tensors="pt")
    
    # Move inputs to the appropriate device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Accumulate frames
    accumulated_frames.append(inputs['pixel_values'])
    
    # Check if duration for accumulating frames is reached
    if time.time() - start_time >= 3:  # 3 seconds
        # Stack accumulated frames along the batch axis
        stacked_frames = torch.cat(accumulated_frames, dim=0)
        
        # Move stacked frames to the appropriate device
        stacked_frames = stacked_frames.to(device)
        
        # Perform inference
        outputs = model(pixel_values=stacked_frames)
        logits = outputs.logits
        # Define an empty list to store all predicted classes
        all_predicted_classes = []

        # Inside the loop where you perform inference:
        # Post-process the inference results to get the predicted class
        predicted_class = torch.argmax(logits, dim=-1)

        # Convert the predicted class tensor to a list and extend the all_predicted_classes list
        all_predicted_classes.extend(predicted_class.tolist())

        # After the loop, find the most common predicted class
        most_common_class, _ = Counter(all_predicted_classes).most_common(1)[0]

        # Convert the most common class to the corresponding label
        sign_language_label = get_sign_language_label(most_common_class)

        
        # Display the results
        cv2.putText(frame, sign_language_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-time Sign Language Detection', frame)
        
        # Reset variables for the next inference
        accumulated_frames = []
        start_time = time.time()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
