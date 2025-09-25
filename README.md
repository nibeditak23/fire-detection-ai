# AI Fire Detection Project

🔥 **Project:** AI Fire Detection using YOLOv8  
📅 **Progress:** Day-2 – Basic fire detection script implemented with test images  

---

## **Project Overview**

This project aims to build an AI-powered **fire and smoke detection system** using computer vision. The system will eventually include:

1. Real-time fire and smoke detection  
2. Fire risk prediction  
3. Automated alerting system  

Currently, the project is at the **basic detection stage**, using **YOLOv8 pre-trained model**. Note that YOLOv8 is **not trained specifically for fire**, so detections may pick up unrelated objects. Fine-tuning on a fire-specific dataset will be implemented in later steps.

---

## **Folder Structure**

fire-detection-ai/
├── data/
│ ├── images/ # Test images for detection
├── src/ # Source code (fire_detector.py)
├── outputs/ # Detection results
├── tests/ # Test scripts
├── docs/ # Project documentation
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions
└── .gitignore # Files/folders to ignore in Git

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
3. Folder Preparation

Ensure you have the following folders and at least one test image in:
data/images/
Running Basic Fire Detection
Test Script
python src/fire_detector.py
This will run YOLOv8 on the test images in data/images/

Output images with bounding boxes will be saved in outputs/

Detected boxes and confidence will be printed in the terminal
Current Notes / Limitations

YOLOv8 is pre-trained on COCO dataset, not fire-specific.

Initial test detections may pick up unrelated objects (e.g., person, pizza) instead of fire.

Fine-tuning with a fire-specific dataset will be implemented in upcoming steps.