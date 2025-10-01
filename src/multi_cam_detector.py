# src/multi_cam_detector.py
from ultralytics import YOLO
import cv2
import os
import json
import threading
from datetime import datetime

class MultiCamDetector:
    def __init__(self, model_path="data/models/best.pt", confidence_threshold=0.25):
        print(f"🔄 Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)  # load normally

        # Patch: handle already-fused models gracefully
        try:
            self.model.fuse()
        except Exception:
            print("⚠️ Skipping fuse (model already fused).")

        self.confidence_threshold = confidence_threshold
        os.makedirs("outputs/videos", exist_ok=True)
        os.makedirs("outputs/logs", exist_ok=True)

        # One log file per day
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = f"outputs/logs/multicam_{date_str}.log"

    def log_detection(self, cam_name, label, conf):
        """Write detection logs with timestamp"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {cam_name}: {label} ({conf:.2f})"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def process_camera(self, cam_name, source):
        """Run detection for one camera/video source"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"⚠️ Could not open source: {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

        out_path = f"outputs/videos/{cam_name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print(f"🎥 Processing {cam_name} ({source}) → {out_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.confidence_threshold)
            result_frame = results[0].plot()

            # Log detections
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                self.log_detection(cam_name, label, conf)

            out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            cv2.imshow(cam_name, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyWindow(cam_name)

    def run_from_config(self, config_path="cameras.json"):
        """Load cameras.json and start threads"""
        with open(config_path, "r") as f:
            cams = json.load(f)

        threads = []
        for cam in cams:
            t = threading.Thread(target=self.process_camera, args=(cam["name"], cam["source"]))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

if __name__ == "__main__":
    detector = MultiCamDetector(model_path="data/models/best.pt")
    detector.run_from_config("cameras.json")
