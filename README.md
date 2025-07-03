# Drone Detection Prototype

This repository contains a prototype developed for a research paper on drone detection using YOLOv5 and Streamlit.

## Overview
- **Purpose:** Detect drones in video streams or files using a YOLOv5 model and visualize results in a Streamlit dashboard.
- **Status:** Prototype (for research purposes only)

## Main Files
- `dashboard.py` — Streamlit dashboard for running detection on webcam or video files.
- `prepare_yolov5_dataset.py` — Script to prepare datasets for YOLOv5 training.
- `requirements.txt` — Python dependencies.
- `yolo11n.pt` — YOLOv5 model weights (included for reproducibility).
- `runs/detect/train4/weights/best.pt` — Best trained model weights from experiments.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```
3. Select input source (webcam or video file) in the sidebar.

## Notes
- Large datasets and videos are **not included** in this repository.
- This code is for research and demonstration purposes only.