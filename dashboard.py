import streamlit as st
import cv2
import os
import glob
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- CONFIG ---
MODEL_PATH = 'runs/detect/train4/weights/best.pt'

# --- Load Model ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# --- Find Video Files ---
def find_videos():
    exts = ('*.mp4', '*.avi', '*.mov')
    files = []
    for ext in exts:
        files.extend(glob.glob(f'**/{ext}', recursive=True))
    return files

video_files = find_videos()

# --- Sidebar ---
st.sidebar.title('Drone Detection Dashboard')
source_type = st.sidebar.radio('Select Input Source:', ['Webcam', 'Video File'])

video_path = None
if source_type == 'Video File' and video_files:
    video_path = st.sidebar.selectbox('Choose a video file:', video_files)

st.sidebar.markdown('---')
st.sidebar.write('Model:', MODEL_PATH)

# --- Main Title ---
st.title('Drone Detection with YOLOv5s')

# --- Video Stream ---
def run_detection(source=0):
    cap = cv2.VideoCapture(source)
    stframe = st.empty()
    stop = False
    if source != 0:
        stop = st.button('Stop Video', key='stop_video')
    while cap.isOpened():
        if source != 0 and stop:
            break
        ret, frame = cap.read()
        if not ret:
            break
        # YOLO expects RGB
        results = model(frame[..., ::-1])
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
        clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"Drone {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        stframe.image(frame, channels='BGR', use_container_width=True)
    cap.release()

# --- Run Detection ---
if source_type == 'Webcam':
    st.info('Using webcam for live detection.')
    run_detection(0)
else:
    if video_path:
        st.info(f'Using video file: {video_path}')
        run_detection(video_path)
    else:
        st.warning('No video files found in the project directory.') 