# Fire Detection AI 🚨🔥

🔥 This project uses **YOLOv8** to detect **fire and smoke** in images, videos, and webcam streams.  
It is designed as the foundation for a **fire safety AI system** with alerting and multi-camera support.

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
│ ├── images/ # Test/demo images
│ ├── models/ # Trained YOLOv8 weights (best.pt here)
│ └── datasets/ # Training datasets (ignored in git)
├── src/
│ └── fire_detector.py # Inference script
├── outputs/ # Saved inference results (ignored in git)
├── tests/ # For future unit tests
├── docs/ # Documentation/notes
├── requirements.txt # Dependencies
└── README.md

---

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

### 🔍 Inference (Local)

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
### 📦 Model
Place your trained YOLOv8 model (best.pt) inside data/models/
Default used in code: data/models/best.pt

### ✅ Current Features
Detect fire & smoke in images, videos, and webcam
Saves annotated results into /outputs
Works with your custom-trained YOLOv8 model

### 📅 Next Steps
 Add multi-feed camera support (Day 5)
 Implement alert system integration (Day 6–7)
 Build dashboard/UI for live monitoring (Day 11–13)
 Plan continuous fine-tuning with new data