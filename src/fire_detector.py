# src/fire_detector.py
from ultralytics import YOLO
import cv2
import os
import argparse
from datetime import datetime


class FireDetector:
    def __init__(self, model_path='data/models/best.pt', confidence_threshold=0.25):
        print(f"🔄 Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    # ---------------- IMAGE ----------------
    def detect_image(self, image_path, save_result=True):
        results = self.model(image_path, conf=self.confidence_threshold)
        result_image = results[0].plot()
        if save_result:
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", f"det_{os.path.basename(image_path)}")
            cv2.imwrite(out_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"✅ Saved: {out_path}")
        return results

    # ---------------- FOLDER ----------------
    def detect_folder(self, folder_path):
        print(f"📂 Running inference on folder: {folder_path}")
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(folder_path, file)
                self.detect_image(img_path)

    # ---------------- VIDEO ----------------
    def detect_video(self, video_path, output_path=None):
        if os.path.isdir(video_path):
            # Process all videos in a folder
            for file in os.listdir(video_path):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.detect_video(os.path.join(video_path, file))
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs("outputs", exist_ok=True)
        if output_path is None:
            # Auto filename with timestamp
            base = os.path.splitext(os.path.basename(video_path))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("outputs", f"det_{base}_{ts}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🎥 Processing video: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=self.confidence_threshold)
            result_frame = results[0].plot()
            out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

            cv2.imshow("Fire Detection Video", cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Processed video saved to: {output_path}")

    # ---------------- WEBCAM ----------------
    def detect_webcam(self, cam_index=0, save_output=False):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("⚠️ Could not open webcam")
            return

        out = None
        if save_output:
            os.makedirs("outputs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("outputs", f"det_webcam_{ts}.avi")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"🎥 Webcam recording enabled → {output_path}")

        print("🎥 Running webcam detection... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=self.confidence_threshold)
            result_frame = results[0].plot()

            if out:
                out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

            cv2.imshow("Fire Detection Webcam", cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["folder", "video", "webcam"], required=True,
                        help="Choose input mode: folder, video, webcam")
    parser.add_argument("--path", type=str, help="Path to folder or video file")
    parser.add_argument("--save-webcam", action="store_true", help="Save webcam output to file")
    args = parser.parse_args()

    detector = FireDetector(model_path="data/models/best.pt")

    if args.mode == "folder":
        detector.detect_folder(args.path)
    elif args.mode == "video":
        detector.detect_video(args.path)
    elif args.mode == "webcam":
        detector.detect_webcam(save_output=args.save_webcam)
