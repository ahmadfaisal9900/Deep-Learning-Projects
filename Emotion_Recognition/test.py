'''
Trying to guess the password here taking the user data as input

Additionally this will take in the name and then iterate through the list of names to check if the name is present in the list or not

Using one hot encoding to convert the names to numbers and then using the model to predict the output

'''




import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import model

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use index 0 for the front camera

# Loop for capturing and processing frames
while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face region to match the input size expected by the model
        resized_face = cv2.resize(face_roi, (48, 48))

        # Convert to tensor and normalize
        tensor = transforms.ToTensor()(resized_face).unsqueeze(0)
        tensor = (tensor - tensor.mean()) / tensor.std()

        # Pass the face through the emotion recognition model
        with torch.no_grad():
            outputs = model(tensor)

        # Get the predicted emotion label
        _, predicted = torch.max(outputs, 1)
        predicted_label = emotion_labels[predicted.item()]

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Overlay the predicted emotion label on the frame
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
