import sys
import os
import random
import math
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QTextEdit, QHBoxLayout, QSizePolicy
)
from PyQt6.QtCore import Qt


# Load LLM (flan-t5-small)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Load MiDaS model and transform
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load YOLOv8
yolo = YOLO("yolov8s.pt")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def midas_depth(frame, scratcher=3000):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_np = np.array(img_rgb)
    result = transform(img_np)
    input_tensor = result["image"].to(device) if isinstance(result, dict) else result.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        elif prediction.dim() == 2:
            prediction = prediction.unsqueeze(0).unsqueeze(0)

        prediction_resized = torch.nn.functional.interpolate(
            prediction,
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False
        )

    depth_map = prediction_resized.squeeze().cpu().numpy()
    inverted_depth = scratcher * (1 / (depth_map + 1e-6))
    return inverted_depth


def depth_to_heatmap(depth_map):
    d_min, d_max = np.min(depth_map), np.max(depth_map)
    norm_depth = (depth_map - d_min) / (d_max - d_min + 1e-8)
    heatmap = (255 * (1 - norm_depth)).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)


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


def generate_summary_llm(car_info):
    if not car_info:
        return "No vehicles detected."

    sorted_info = sorted(enumerate(car_info, start=1), key=lambda x: x[1][2])
    prompt_lines = []

    for i, (idx, (cx, cy, dist)) in enumerate(sorted_info):
        prompt_lines.append(f"Vehicle {idx} is at ({cx},{cy}) approximately {dist:.1f} meters away.")

    for i in range(len(sorted_info) - 1):
        idx1, (_, _, d1) = sorted_info[i]
        idx2, (_, _, d2) = sorted_info[i + 1]
        spacing = np.sqrt((d1 - d2)**2 + 4)
        prompt_lines.append(f"Vehicle {idx1} and {idx2} are {spacing:.1f} meters apart.")

    full_prompt = "Summarize this traffic scene:\n" + "\n".join(prompt_lines)
    inputs = tokenizer("Summarize: " + full_prompt, return_tensors="pt", truncation=True)
    outputs = llm.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def plot_results(frame, depth_heatmap, yolo_annotated):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Frame")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB))
    axs[1].set_title("MiDaS Depth Heatmap")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB))
    axs[2].set_title("YOLOv8 + Depth + Distances")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


class VideoAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Analyzer with LLM")
        self.setGeometry(100, 100, 700, 500)

        self.layout = QVBoxLayout()

        # File selector
        hlayout = QHBoxLayout()
        self.label = QLabel("No video selected")
        hlayout.addWidget(self.label)

        self.btn_browse = QPushButton("Browse Video")
        self.btn_browse.clicked.connect(self.browse_file)
        hlayout.addWidget(self.btn_browse)

        self.layout.addLayout(hlayout)

        # Run button
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.btn_run)

        # Text box for LLM summary
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.output)

        self.setLayout(self.layout)
        self.video_path = None

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "MP4 Files (*.mp4);;All Files (*)")
        if file_path:
            self.video_path = file_path
            self.label.setText(f"Selected: {file_path}")
            self.output.clear()

    def run_analysis(self):
        if not self.video_path or not os.path.isfile(self.video_path):
            self.output.setPlainText("Please select a valid video file first.")
            return

        self.output.setPlainText("Processing... Please wait.")
        QApplication.processEvents()

        summary = self.analyze_video(self.video_path)
        self.output.setPlainText(summary)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return "Video has no frames or cannot be read."

        selected_frame_idx = random.randint(0, frame_count - 1)

        current_frame = 0
        summary = "No vehicles detected."
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame == selected_frame_idx:
                depth_map = midas_depth(frame)
                depth_heatmap = depth_to_heatmap(depth_map)
                yolo_results = yolo(frame)
                annotated_yolo, car_info = annotate_with_yolo(frame, yolo_results, depth_map)

                # Add pairwise distance lines & labels
                annotated_yolo = annotate_pairwise_depth_distances(annotated_yolo, car_info)

                plot_results(frame, depth_heatmap, annotated_yolo)

                summary = generate_summary_llm(car_info)
                break
            current_frame += 1

        cap.release()
        return summary


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec())
