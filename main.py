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
from depth_utils import midas, midas_depth, depth_to_heatmap
from yolo import annotate_pairwise_depth_distances, annotate_with_yolo, yolo
from llm import generate_summary_llm
from road import random_frame_predictions, model

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
                # Step 1: Depth Estimation
                depth_map = midas_depth(frame)
                depth_heatmap = depth_to_heatmap(depth_map)

                # Step 2: YOLOv8 detections & car analysis
                yolo_results = yolo(frame)
                annotated_yolo, car_info = annotate_with_yolo(frame, yolo_results, depth_map)
                annotated_yolo = annotate_pairwise_depth_distances(annotated_yolo, car_info)

                # Step 3: Road segmentation (overlay mask)
                try:
                    road_result = random_frame_predictions(video_path=video_path, model=model)
                    road_mask = road_result.get('mask')

                    if road_mask is not None:
                        dark_purple = (80, 0, 80)
                        mask_bool = road_mask.astype(bool)
                        overlay = annotated_yolo.copy()
                        overlay[mask_bool] = dark_purple
                        combined_frame = cv2.addWeighted(annotated_yolo, 0.7, overlay, 0.3, 0)
                    else:
                        print("[WARN] No 'mask' returned from road prediction.")
                        combined_frame = annotated_yolo
                except Exception as e:
                    print(f"[ERROR] Failed to apply road mask: {e}")
                    combined_frame = annotated_yolo

                # Step 4: Visualization
                plot_results(frame, depth_heatmap, combined_frame)

                # Step 5: Natural language summary
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
