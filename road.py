import cv2
import matplotlib.pyplot as plt
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="mq4xwNHHmCOLigA78VzE")

# Load your project by name and version
project = rf.workspace("jk-nanu0").project("final2-xiiin")
model = project.version(2).model

def show_prediction_on_frame(frame_path, prediction):
    # Load image with OpenCV
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Draw boxes
    for obj in prediction['predictions']:
        # Example keys, adjust to your actual prediction keys
        x1, y1 = int(obj['x'] - obj['width'] / 2), int(obj['y'] - obj['height'] / 2)
        x2, y2 = int(obj['x'] + obj['width'] / 2), int(obj['y'] + obj['height'] / 2)
        label = obj['class']
        conf = obj['confidence']

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        # Put label text
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


import cv2
import random

def random_frame_predictions(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frame_idx = random.randint(0, frame_count - 1)

    current_frame = 0
    frame = None
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if current_frame == selected_frame_idx:
            frame = f
            break
        current_frame += 1

    cap.release()
    if frame is None:
        return None

    frame_path = "frame.jpg"
    cv2.imwrite(frame_path, frame)

    prediction = model.predict(frame_path).json()

    # Combine all masks if present
    h, w, _ = frame.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for obj in prediction.get("predictions", []):
        if "mask" in obj:
            mask = np.array(obj["mask"], dtype=np.uint8)
            mask = cv2.resize(mask, (w, h))
            combined_mask = np.maximum(combined_mask, mask)

    return {
        "frame": frame,
        "mask": combined_mask,
        "raw": prediction
    }
