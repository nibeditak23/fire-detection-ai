# Fire Detection AI 🚨🔥

🔥 An AI-powered fire & smoke detection system using YOLOv8.  
Supports inference on **images, videos, webcam feeds, and multiple camera sources** (via `multi_cam_detector.py`).  

---

## **Project Overview**

This project aims to build an AI-powered **fire and smoke detection system** using computer vision. The system will eventually include:

1. Real-time fire and smoke detection  
2. Fire risk prediction  
3. Automated alerting system  


---

## **Folder Structure**

fire-detection-ai/
├── data/
│ ├── images/ # Test images for inference
│ ├── videos/ # Test/demo videos
│ ├── models/ # Trained models (best.pt etc.)
│ └── datasets/ # Training datasets (not in repo, stored in Drive)
├── src/
│ ├── fire_detector.py # Single feed inference (image/video/webcam/folder)
│ ├── multi_cam_detector.py # Multi-camera real-time monitoring
├── outputs/ # Saved inference results (images/videos/logs)
├── tests/ # Unit tests (future)
├── docs/ # Documentation
├── requirements.txt # Dependencies
├── README.md # Project overview
└── .gitignore

---
## 🚀 Features

- Detects **fire** and **smoke** using YOLOv8.
- Works with:
  - Images (`--mode folder --path data/images`)
  - Videos (`--mode video --path data/videos/sample.mp4`)
  - Webcam (`--mode webcam`)
  - Multiple cameras (`multi_cam_detector.py` + `cameras.json`)
- Saves annotated results in `outputs/`
- Logs detections with **camera name + timestamp**.

## **Setup Instructions**

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fire-detection-ai.git
cd fire-detection-ai
2. Set up Virtual Environment
python -m venv fire-env
# Activate virtual environment
fire-env\Scripts\activate      # Windows
# Install dependencies
pip install -r requirements.txt

### 🏋️ Training (in Colab)
We trained the model in Google Colab using fire_smoke.zip dataset.
# Steps:
Upload fire_smoke.zip to Drive (/MyDrive/fire_detection/datasets/).
Run the provided Colab notebook (fire_yolov8_day4_xxx.ipynb).
Training will generate weights:
best.pt → best-performing weights
last.pt → last saved weights
Save best.pt into:
data/models/best.pt

### 📦 Usage - fire_detector.py
## 🔹 Single feed detection

# 1. Run detection on a single image:
python src/fire_detector.py --mode single --path data/images/fire18.jpg
# 2. Run on all images in a folder:
python src/fire_detector.py --mode folder --path data/images
# 3. Run inference on a video file
python src/fire_detector.py --mode video --path data/videos/test_fire.mp4
# 4. Detect all videos in a folder
python src/fire_detector.py --mode video --path data/videos/
# 5. Run inference with webcam
python src/fire_detector.py --mode webcam
# 6. Webcam + Save recording
python src/fire_detector.py --mode webcam --save-webcam
# note - for videos and webcam - Press q to quit
# Annotated results are saved in:
outputs/
Example:
fire18.jpg → 2 detections (2 fires)

## 🔹 Multi-camera detection
Configure your sources in cameras.json:(test)
[
  { "name": "Parking-Cam", "source": "data/videos/test_fire.mp4" },
  { "name": "Warehouse-Cam", "source": "data/videos/test_smoke.mp4" }
]
Prod like:
[
  { "name": "Lobby-Cam", "source": "rtsp://192.168.1.10:554/live" },
  { "name": "Parking-Cam", "source": "rtsp://192.168.1.11:554/live" },
  { "name": "Server-Room-Cam", "source": "0" }   // local webcam
]
Run:
python src/multi_cam_detector.py
# 📊 Outputs
Annotated images → outputs/
Annotated videos → outputs/videos/
Logs → outputs/multicam_YYYY-MM-DD.log

### 📦 Model
Place your trained YOLOv8 model (best.pt) inside data/models/
Default used in code: data/models/best.pt

### ✅ Current Features
Detect fire & smoke in images, videos, and webcam
Saves annotated results into /outputs
Works with your custom-trained YOLOv8 model
multi-feed camera support 

### 📅 Next Steps
 Implement alert system integration (Day 6–7)
 Build dashboard/UI for live monitoring (Day 11–13)
 Plan continuous fine-tuning with new data