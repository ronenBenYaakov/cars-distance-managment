from ultralytics import YOLO
import cv2
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

yolo = YOLO("yolov8s.pt")

def annotate_with_yolo(frame, yolo_results, depth_map, gamma=2):
    annotated = frame.copy()
    car_info = []

    for result in yolo_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls not in [2, 7]:  # Only 'car' and 'truck' classes (COCO)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = np.clip(cx, 0, depth_map.shape[1] - 1)
            cy = np.clip(cy, 0, depth_map.shape[0] - 1)
            distance = depth_map[cy, cx]
            distance *= gamma * sigmoid(distance)
            car_info.append((cx, cy, distance))

            label = f"{distance:.2f}m"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, car_info


def annotate_pairwise_depth_distances(image, car_info):
    annotated = image.copy()

    # Sort cars by distance (nearest first)
    sorted_info = sorted(car_info, key=lambda x: x[2])  # (cx, cy, dist)

    for i in range(len(sorted_info) - 1):
        x1, y1, d1 = sorted_info[i]
        x2, y2, d2 = sorted_info[i + 1]

        # Calculate pairwise distance (Euclidean approx)
        distance_diff = np.sqrt((d1 - d2) ** 2 + 4)  # +4 to avoid zero

        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # Draw line and distance label
        cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(annotated, f"{distance_diff:.2f}m", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return annotated