import cv2
import os
import json
import threading
import datetime
from ultralytics import YOLO
import pygame


class MultiCamDetector:
    def __init__(self, model_path="data/models/best.pt", confidence_threshold=0.25):
        print(f"🔄 Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)  # YOLOv8 model
        self.confidence_threshold = confidence_threshold
        self.model_lock = threading.Lock()  # lock to avoid GPU/CPU conflicts

        # 🔔 Setup siren
        pygame.mixer.init()
        self.siren_playing = False

        # 📝 Setup logging
        os.makedirs("outputs/logs", exist_ok=True)
        log_file = datetime.datetime.now().strftime("outputs/logs/multicam_%Y%m%d_%H%M.log")
        self.log_fh = open(log_file, "a")

    # ---------------- Siren ----------------
    def play_siren(self, siren_path="data/siren/fire-alarm.mp3"):
        if not self.siren_playing:
            try:
                pygame.mixer.music.load(siren_path)
                pygame.mixer.music.play(-1)  # loop until stopped
                self.siren_playing = True
                print("🚨 Siren started!")
            except Exception as e:
                print("⚠️ Could not play siren:", e)

    def stop_siren(self):
        if self.siren_playing:
            pygame.mixer.music.stop()
            self.siren_playing = False
            print("✅ Siren stopped.")

    # ---------------- Logging ----------------
    def log_detection(self, cam_name, label, conf):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {cam_name}: {label} ({conf:.2f})"
        print(line)
        self.log_fh.write(line + "\n")
        self.log_fh.flush()

    # ---------------- Process one camera ----------------
    def process_camera(self, cam_name, source, stop_event):
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"⚠️ Could not open source for {cam_name}: {source}")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

            os.makedirs("outputs/videos", exist_ok=True)
            out_path = f"outputs/videos/{cam_name}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            print(f"🎥 Processing {cam_name} ({source}) → {out_path}")

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                with self.model_lock:
                    results = self.model(frame, conf=self.confidence_threshold)

                result_img = results[0].plot()
                out.write(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                cv2.imshow(cam_name, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                # 🔎 Check detections
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results[0].names[cls_id]
                    self.log_detection(cam_name, label, conf)

                    if label in ["fire", "smoke"] and conf > 0.3:
                        self.play_siren()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()  # tell all threads to stop

            cap.release()
            out.release()
            cv2.destroyWindow(cam_name)
            print(f"✅ Finished {cam_name} processing.")

        except Exception as e:
            print(f"⚠️ Error in camera {cam_name}: {e}")

    # ---------------- Multi-Camera Runner ----------------
    def run_from_config(self, config_file="cameras.json"):
        if not os.path.exists(config_file):
            print(f"❌ Config not found: {config_file}")
            return

        with open(config_file, "r") as fh:
            cameras = json.load(fh)

        stop_event = threading.Event()
        threads = []
        for cam in cameras:
            name = cam.get("name") or f"cam_{len(threads)+1}"
            source = cam.get("source")
            t = threading.Thread(
                target=self.process_camera, args=(name, source, stop_event), daemon=True
            )
            t.start()
            threads.append(t)

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            self.stop_siren()
            self.log_fh.close()
            cv2.destroyAllWindows()
            print("🛑 Shutdown complete.")


if __name__ == "__main__":
    detector = MultiCamDetector(model_path="data/models/best.pt")
    detector.run_from_config("cameras.json")
