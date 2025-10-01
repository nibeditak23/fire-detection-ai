# src/fire_detector.py
from ultralytics import YOLO
import cv2
import os
import argparse
from datetime import datetime
import pygame


class FireDetector:
    def __init__(self, model_path='data/models/best.pt', confidence_threshold=0.25, alert_frames=3):
        print(f"🔄 Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.alert_frames = alert_frames  # consecutive frames to trigger siren

        # create output folder
        os.makedirs("outputs", exist_ok=True)

        # initialize pygame mixer for siren
        pygame.mixer.init()

    # ---------------- siren ----------------
    def play_siren(self, siren_path="data/sirem/fire-alarm.mp3"):
        try:
            pygame.mixer.music.load(siren_path)
            pygame.mixer.music.play()
        except Exception as e:
            print("⚠️ Could not play siren:", e)

    # ---------------- VIDEO ----------------
    def detect_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)

        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("outputs", f"det_{base}_{ts}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🎥 Processing video: {video_path}")
        fire_counter = 0  # counts consecutive frames with fire/smoke

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
            result_frame = results[0].plot()

            num_boxes = len(results[0].boxes) if results[0].boxes is not None else 0
            if num_boxes > 0:
                fire_counter += 1
            else:
                fire_counter = 0

            # Trigger siren if fire detected in consecutive frames
            if fire_counter >= self.alert_frames:
                print(f"🚨 FIRE/SMOKE DETECTED in {os.path.basename(video_path)}! 🚨")
                self.play_siren("data/siren/fire-alarm.mp3")
                fire_counter = 0  # reset counter to avoid continuous alarm

            out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            cv2.imshow("Fire Detection Video", cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Processed video saved to: {output_path}")

    # ---------------- FOLDER OF VIDEOS ----------------
    def detect_videos_in_folder(self, folder_path):
        video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        for video in video_files:
            self.detect_video(video)


# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection in Video Folder with Siren Alert")
    parser.add_argument("--path", type=str, required=True, help="Path to folder containing videos")
    parser.add_argument("--model", type=str, default="data/models/best.pt", help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--alert-frames", type=int, default=3, help="Consecutive frames to trigger siren")
    args = parser.parse_args()

    detector = FireDetector(model_path=args.model, confidence_threshold=args.conf, alert_frames=args.alert_frames)
    detector.detect_videos_in_folder(args.path)
