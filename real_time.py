import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from torchvision.models import resnet18

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=False)
num_classes = 8
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('pre_trained_model.pt', map_location=device))
model.to(device)
model.eval()

# Define the transformation
transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define class labels
classes = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

# Function to make predictions on a frame
# Function to make predictions on a frame
def predict_frame(frame, confidence_threshold=0.7):
    img = Image.fromarray(frame)
    img = transformation(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        max_confidence, predicted = torch.max(probabilities, dim=0)

    # Check if the confidence exceeds the threshold
    if max_confidence.item() >= confidence_threshold:
        class_name = classes[predicted.item()]
        return class_name, max_confidence.item()
    else:
        return None, None

# Start video capture from the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

# Adjusted main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Make prediction
    class_name, confidence = predict_frame(frame)

    if class_name:  # Only display predictions with sufficient confidence
        cv2.putText(frame, f"Detected: {class_name} ({confidence:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with OpenCV
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
