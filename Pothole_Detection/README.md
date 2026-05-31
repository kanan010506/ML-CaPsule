# 🕳️ Pothole Detection using YOLOv8

## 📌 Overview
This project fine-tunes a YOLOv8 object detection model to detect potholes
in real-time from road images and dashcam video footage. It outputs bounding
boxes with confidence scores overlaid on the original frame, making it useful
for infrastructure monitoring and road safety applications.

This is especially relevant in the Indian context where pothole-related
accidents remain a significant public safety concern.

## 🧠 Model
- **Architecture**: YOLOv8n (nano) — fine-tuned on pothole dataset
- **Framework**: Ultralytics YOLOv8
- **Task**: Object Detection (single class: `pothole`)

## 🗂️ Dataset
- **Source**: [Pothole Detection Dataset — Roboflow Universe](https://universe.roboflow.com/roboflow-100/pothole-detection-dataset)
- **Format**: YOLOv8 (images + labels in `.txt`)
- **Split**: 80% Train / 10% Val / 10% Test
- **Classes**: 1 (`pothole`)

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| YOLOv8 (Ultralytics) | Object detection model |
| OpenCV | Image & video processing |
| NumPy | Array operations |
| Matplotlib | Visualization |
| Google Colab | Training environment |

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install ultralytics opencv-python matplotlib
```

### 2. Clone and open notebook
```bash
git clone https://github.com/Niketkumardheeryan/ML-CaPsule.git
cd ML-CaPsule/Pothole_Detection_YOLOv8
jupyter notebook Pothole_Detection_YOLOv8.ipynb
```

### 3. Run all cells
The notebook will download the dataset, train the model, and run inference
with bounding box visualization.

## 📊 Results
| Metric | Value |
|--------|-------|
| mAP@50 | ~0.87 |
| Precision | ~0.84 |
| Recall | ~0.81 |

> Results may vary based on dataset version and training epochs.

## 📁 Project Structure
Pothole_Detection_YOLOv8/
├── Pothole_Detection_YOLOv8.ipynb   # Main notebook
└── README.md                         # Project documentation

## 🔍 Alternatives Considered
A classical image processing approach using Canny edge detection and contour
analysis was explored, but YOLOv8 significantly outperforms it in accuracy,
generalization, and real-time inference speed — making it the right choice
for production-ready road safety applications.

## 🌐 Contributing Under
GSSoC '26

## 📄 License
MIT License
