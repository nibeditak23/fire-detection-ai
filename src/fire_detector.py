# src/fire_detector.py
from ultralytics import YOLO
import cv2
import os

class FireDetector:
    def __init__(self, model_path='yolov8s.pt', confidence_threshold=0.25):
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)  # Download pre-trained YOLOv8
        self.confidence_threshold = confidence_threshold

    def detect_fire(self, image_path, save_result=True, show_result=False):
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            return []

        results = self.model(image_path, conf=self.confidence_threshold)

        # Prepare output path
        os.makedirs("outputs", exist_ok=True)
        base_name = os.path.basename(image_path)
        output_image_path = os.path.join("outputs", f"det_{base_name}")

        # Draw bounding boxes
        result_image = results[0].plot()

        if save_result:
            cv2.imwrite(output_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"✅ Result saved to: {output_image_path}")

        if show_result:
            cv2.imshow("Fire Detection", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()

        # Return bounding boxes as list
        return results[0].boxes.xyxy.tolist() if results[0].boxes is not None else []

# Quick test: process all images in data/images
if __name__ == "__main__":
    detector = FireDetector()
    image_folder = "data/images"

    for file in os.listdir(image_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(image_folder, file)
            boxes = detector.detect_fire(img_path)
            print(f"{file} → Detected boxes: {boxes}")
