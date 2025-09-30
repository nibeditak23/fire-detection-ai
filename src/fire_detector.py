# src/fire_detector.py
from ultralytics import YOLO
import cv2
import os
import glob
import argparse


class FireDetector:
    def __init__(self, model_path="data/models/best.pt", confidence_threshold=0.25):
        print(f"🔄 Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_image(self, image_path, save_result=True, show_result=False):
        """Run detection on a single image"""
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            return []

        results = self.model.predict(image_path, conf=self.confidence_threshold)

        # Ensure outputs/ folder exists
        os.makedirs("outputs", exist_ok=True)

        base_name = os.path.basename(image_path)
        output_path = os.path.join("outputs", f"det_{base_name}")

        # Annotated image
        result_img = results[0].plot()

        if save_result:
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"✅ Saved: {output_path}")

        if show_result:
            cv2.imshow("Fire Detection", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        return results[0].boxes.xyxy.tolist() if results[0].boxes is not None else []

    def detect_folder(self, folder_path):
        """Run detection on all images in a folder"""
        print(f"📂 Running inference on folder: {folder_path}")
        img_files = glob.glob(os.path.join(folder_path, "*.*"))
        img_files = [f for f in img_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img in img_files:
            boxes = self.detect_image(img, save_result=True, show_result=False)
            print(f"{os.path.basename(img)} → {len(boxes)} detections")

    def detect_video(self, video_source=0, save_output=False):
        """Run real-time detection from webcam or video file"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("⚠️ Cannot open video source")
            return

        out = None
        if save_output and isinstance(video_source, str):
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", "det_video.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=self.confidence_threshold)
            result_frame = results[0].plot()

            if out:
                out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

            cv2.imshow("Fire Detection - Video", cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if out:
            out.release()
            print(f"✅ Video saved to outputs/det_video.avi")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection using YOLOv8")
    parser.add_argument("--mode", type=str, choices=["image", "folder", "video"], required=True,
                        help="Run mode: image | folder | video")
    parser.add_argument("--path", type=str, default="data/images/sample_fire.jpg",
                        help="Path to image, folder, or video")
    parser.add_argument("--model", type=str, default="data/models/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    detector = FireDetector(model_path=args.model, confidence_threshold=args.conf)

    if args.mode == "image":
        detector.detect_image(args.path, save_result=True, show_result=False)
    elif args.mode == "folder":
        detector.detect_folder(args.path)
    elif args.mode == "video":
        detector.detect_video(args.path, save_output=True)
