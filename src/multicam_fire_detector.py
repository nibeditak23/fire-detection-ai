# src/multi_cam_detector.py
from ultralytics import YOLO
import cv2
import os
import sys
import json
import threading
import datetime
import pygame
import uuid, hashlib
from dotenv import load_dotenv
from twilio.rest import Client
import time
import pyttsx3
import queue
import socket
import tempfile
import platform
from pathlib import Path

# ---------------- HWID ----------------
def get_hardware_id():
    mac = uuid.getnode()
    system_info = f"{mac}-{os.getenv('COMPUTERNAME', '')}"
    return hashlib.sha256(system_info.encode()).hexdigest()

# ---------------- License ----------------
def validate_license(license_file="license.lic"):
    if not os.path.exists(license_file):
        print(f"❌ License file '{license_file}' not found!")
        sys.exit(1)

    with open(license_file, "r") as f:
        licensed_hwid = f.read().strip()

    current_hwid = get_hardware_id()
    if licensed_hwid != current_hwid:
        print("❌ Invalid License! This exe is not authorized for this machine.")
        sys.exit(1)
    else:
        print("✅ License validated. Running application...")

# ---------------- Resource Path ----------------
def resource_path(relative_path):
    """ Get absolute path for PyInstaller and normal runs """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- Fire Detector ----------------
class MultiCamDetector:
    def __init__(self, model_path=None, confidence_threshold=0.25):
        # Model path resolution
        if model_path is None:
            model_path = resource_path("data/models/best.pt")
        else:
            model_path = resource_path(model_path)

        print(f"🔄 Loading YOLOv8 model from {model_path} ...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.model_lock = threading.Lock()

        # Initialize pygame mixer for siren and audio playback
        pygame.mixer.init()
        self.siren_playing = False

        # Logging
        os.makedirs("outputs/logs", exist_ok=True)
        log_file = datetime.datetime.now().strftime("outputs/logs/multicam_%Y%m%d_%H%M.log")
        self.log_fh = open(log_file, "a")

        # ------------------- SMS Setup -------------------
        load_dotenv()
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_PHONE_NUMBER")
        self.alert_numbers = []

        if self.twilio_sid and self.twilio_token:
            try:
                self.twilio_client = Client(self.twilio_sid, self.twilio_token)
                print("📲 Twilio client initialized.")
            except Exception as e:
                print("⚠️ Twilio init failed:", e)
                self.twilio_client = None
        else:
            print("⚠️ Twilio credentials not found in .env — SMS alerts disabled.")
            self.twilio_client = None

        self.last_sms_time = {}
        self.sms_cooldown = 60  # seconds between SMS for same camera

        # >>> [VOICE ALERT ADDITION START] <<<
        # Voice configuration (all configurable via .env)
        self.voice_enabled = os.getenv("VOICE_ALERT_ENABLED", "true").lower() in ("1", "true", "yes")
        self.voice_repeat_interval = int(os.getenv("VOICE_REPEAT_INTERVAL", "10"))  # seconds between repeats
        self.tts_rate = int(os.getenv("VOICE_RATE", "160"))                         # speech speed
        self.voice_cooldown = int(os.getenv("VOICE_COOLDOWN", "10"))                # seconds between re-triggers

        # Per-camera state
        self.last_voice_time = {}
        self.voice_threads = {}

        # Shared voice queue and worker (single audio engine worker)
        self.voice_queue = queue.Queue()
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        # >>> [VOICE ALERT ADDITION END] <<<

    # ---------------- Siren ----------------
    def play_siren(self, siren_path=None):
        if siren_path is None:
            siren_path = resource_path("data/siren/fire-alarm.mp3")
        else:
            siren_path = resource_path(siren_path)

        if not self.siren_playing:
            try:
                pygame.mixer.music.load(siren_path)
                pygame.mixer.music.play(-1)
                self.siren_playing = True
                print("🚨 Siren started!")
            except Exception as e:
                print("⚠️ Could not play siren:", e)

    def stop_siren(self):
        if self.siren_playing:
            try:
                pygame.mixer.music.stop()
            except Exception as e:
                print("⚠️ Error stopping siren:", e)
            self.siren_playing = False
            print("✅ Siren stopped.")

    # ---------------- Voice queueing (public) ----------------
    def speak_alert_once(self, location):
        """Queue a message for the voice worker to speak safely."""
        if not self.voice_enabled:
            return
        message = f"Attention. Fire detected in {location}. Please evacuate immediately."
        # log queued
        self.log_fh.write(f"[VOICE QUEUED] {datetime.datetime.now()} - {location} - {message}\n")
        self.log_fh.flush()
        self.voice_queue.put((location, message))

    # ---------------- Voice helpers (cross-platform & hybrid) ----------------
    @staticmethod
    def _slugify(text: str) -> str:
        """Simple filename-safe slug."""
        return "".join(c if c.isalnum() else "_" for c in text).lower()

    def _get_voice_cache_path(self, location: str) -> str:
        """Return path to cached mp3 for location."""
        cache_dir = resource_path("data/voice_cache")
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"{self._slugify(location)}.mp3"
        return os.path.join(cache_dir, fname)

    def _is_online(self, timeout: float = 2.0) -> bool:
        """Quick internet check (returns True if DNS reachable)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(("8.8.8.8", 53))
            sock.close()
            return True
        except Exception:
            return False

    def _generate_cached_audio(self, location: str, message: str) -> bool:
        """
        Use gTTS to create a cached mp3 for `location`. Returns True if successful.
        Requires internet and gTTS installed.
        """
        path = self._get_voice_cache_path(location)
        try:
            from gtts import gTTS
        except Exception:
            print("⚠️ gTTS not installed; cannot generate cached audio.")
            return False

        try:
            tts = gTTS(text=message, lang="en")
            tts.save(path)
            print(f"✅ Cached TTS saved: {path}")
            return True
        except Exception as e:
            print(f"⚠️ gTTS generation failed for {location}: {e}")
            return False

    def _play_cached_audio(self, path: str) -> bool:
        """Play mp3/wav via pygame mixer so it shares the same audio device as siren.
           Blocks until the sound finishes to keep messages sequential."""
        try:
            snd = pygame.mixer.Sound(path)
            ch = snd.play()
            if ch is None:
                # On some systems a Channel may not be returned immediately
                time.sleep(0.1)
                ch = snd.play()
            # block until finished
            while ch.get_busy():
                time.sleep(0.08)
            return True
        except Exception as e:
            print(f"⚠️ play_cached_audio failed ({path}): {e}")
            return False

    def _play_with_pyttsx3(self, message: str) -> bool:
        """
        Fallback: use pyttsx3 to speak. On Windows we pause/unpause the siren to
        reduce audio device contention (best-effort).
        """
        try:
            if self.siren_playing:
                try:
                    pygame.mixer.music.pause()
                except Exception:
                    pass
            engine = pyttsx3.init()
            engine.setProperty("rate", self.tts_rate)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception as e:
            print(f"⚠️ pyttsx3 speak failed: {e}")
            return False
        finally:
            if self.siren_playing:
                try:
                    pygame.mixer.music.unpause()
                except Exception:
                    pass

    def _play_message(self, location: str, message: str):
        """
        Best-effort play routine:
         1) if cached file exists -> play it (recommended)
         2) else if online -> generate cache with gTTS then play
         3) else fallback to pyttsx3 (Linux reliable) or pause/unpause trick on Windows.
        """
        cache_path = self._get_voice_cache_path(location)

        # 1) Cached file
        if os.path.exists(cache_path):
            ok = self._play_cached_audio(cache_path)
            if ok:
                return

        # 2) Try to create cache (online)
        if self._is_online():
            ok = self._generate_cached_audio(location, message)
            if ok and os.path.exists(cache_path):
                ok2 = self._play_cached_audio(cache_path)
                if ok2:
                    return

        # 3) Offline fallback - prefer pyttsx3 on Linux/Jetson
        if platform.system().lower().startswith("linux"):
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.tts_rate)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
                return
            except Exception as e:
                print(f"⚠️ Offline pyttsx3 failed on Linux: {e}")

        # Final fallback (Windows best-effort)
        self._play_with_pyttsx3(message)

    # ---------------- Voice queue worker ----------------
    def _voice_worker(self):
        """Dedicated background thread to handle all text-to-speech safely."""
        print("🎤 Voice worker started (shared audio approach).")
        while True:
            try:
                location, message = self.voice_queue.get()
                if message is None:
                    # shutdown signal
                    break
                # Log and play
                self.log_fh.write(f"[VOICE PLAY] {datetime.datetime.now()} - {location} - {message}\n")
                self.log_fh.flush()
                self._play_message(location, message)
                self.voice_queue.task_done()
                time.sleep(0.12)  # brief breathing gap
            except Exception as e:
                print(f"⚠️ Voice worker error: {e}")
                time.sleep(0.2)

    # ---------------- Combined alert (siren + per-camera voice loop) ----------------
    def trigger_combined_alert(self, location):
        """Play global siren and repeat voice alert for this specific camera."""
        now = time.time()
        last_voice = self.last_voice_time.get(location, 0)

        # Cooldown before creating a new voice loop for same camera
        if now - last_voice < self.voice_cooldown:
            print(f"🕒 Voice cooldown active for {location}, skipping new loop.")
            return
        self.last_voice_time[location] = now

        # Ensure siren is running
        if not self.siren_playing:
            self.play_siren()

        # If this camera already has an active voice-loop thread, skip starting another
        if location in self.voice_threads and self.voice_threads[location].is_alive():
            print(f"ℹ️ Voice loop already running for {location}")
            return

        # Per-camera loop: enqueue message every voice_repeat_interval while siren plays
        def voice_loop(loc):
            print(f"🗣️ Voice loop started for {loc}")
            while self.siren_playing:
                # queue one message for the worker to play
                self.speak_alert_once(loc)
                time.sleep(self.voice_repeat_interval)
            print(f"🛑 Voice loop stopped for {loc}")

        t = threading.Thread(target=voice_loop, args=(location,), daemon=True)
        t.start()
        self.voice_threads[location] = t

    # ---------------- Logging ----------------
    def log_detection(self, cam_name, label, conf):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {cam_name}: {label} ({conf:.2f})"
        print(line)
        self.log_fh.write(line + "\n")
        self.log_fh.flush()

    # ---------------- SMS Alerts ----------------
    def send_sms_alert(self, cam_name, label, conf):
        if not self.twilio_client:
            return

        current_time = time.time()
        last_sent = self.last_sms_time.get(cam_name, 0)
        if current_time - last_sent < self.sms_cooldown:
            cooldown_remaining = int(self.sms_cooldown - (current_time - last_sent))
            msg = (f"[INFO] SMS skipped for {cam_name} "
                   f"({label}, conf={conf:.2f}) - cooldown active ({cooldown_remaining}s left)")
            print(msg)
            self.log_fh.write(msg + "\n")
            self.log_fh.flush()
            return

        msg_body = f"🚨 ALERT: {label.upper()} detected on {cam_name} (confidence {conf:.2f})"
        for to_number in self.alert_numbers:
            try:
                self.twilio_client.messages.create(
                    body=msg_body,
                    from_=self.twilio_from,
                    to=to_number
                )
                print(f"📲 SMS alert sent to {to_number}: {msg_body}")
            except Exception as e:
                print(f"⚠️ Failed to send SMS to {to_number}: {e}")

        self.last_sms_time[cam_name] = current_time

    # ---------------- Process Camera ----------------
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

                fire_detected = False
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results[0].names[cls_id]
                    self.log_detection(cam_name, label, conf)

                    if label in ["fire", "smoke"] and conf > 0.3:
                        fire_detected = True
                        # Start combined alert (siren + voice loop for this camera)
                        self.trigger_combined_alert(cam_name)
                        # SMS (unchanged)
                        self.send_sms_alert(cam_name, label, conf)

                # Stop siren only when no camera has active voice-loop thread
                if not fire_detected:
                    active_voices = any(th.is_alive() for th in self.voice_threads.values())
                    if not active_voices and self.siren_playing:
                        print(f"✅ No active fires — stopping siren and voice loops.")
                        self.siren_playing = False
                        self.stop_siren()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()

            cap.release()
            out.release()
            cv2.destroyWindow(cam_name)
            print(f"✅ Finished {cam_name} processing.")
        except Exception as e:
            print(f"⚠️ Error in camera {cam_name}: {e}")

    # ---------------- Run from Config ----------------
    def run_from_config(self, config_file="cameras.json"):
        if not os.path.exists(config_file):
            print(f"❌ Config not found: {config_file}")
            return

        with open(config_file, "r") as fh:
            cameras = json.load(fh)

        alerts = cameras.get("alerts", {})
        self.alert_numbers = alerts.get("recipients", [])
        print(f"📱 Alert recipients loaded: {self.alert_numbers}")

        cameras = cameras.get("cameras", [])
        stop_event = threading.Event()
        threads = []

        for cam in cameras:
            name = cam.get("name") or f"cam_{len(threads)+1}"
            source = cam.get("source")
            t = threading.Thread(target=self.process_camera, args=(name, source, stop_event), daemon=True)
            t.start()
            threads.append(t)

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            # Stop everything cleanly
            self.siren_playing = False
            self.stop_siren()
            # Signal voice worker to exit
            try:
                self.voice_queue.put((None, None))
            except Exception:
                pass
            self.log_fh.close()
            cv2.destroyAllWindows()
            print("🛑 Shutdown complete.")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Camera Fire/Smoke Detection with Siren + Voice Alerts")
    parser.add_argument("--config", type=str, help="Path to JSON config file containing camera sources")
    parser.add_argument("--model", type=str, default="data/models/best.pt", help="Path to trained YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--get-hwid", action="store_true", help="Generate and print hardware ID")
    parser.add_argument("--license", type=str, default="license.lic", help="Path to license file")
    args = parser.parse_args()

    if args.get_hwid:
        hwid = get_hardware_id()
        print("Your Hardware ID:", hwid)
        with open("hwid.txt", "w") as f:
            f.write(hwid)
        print("✅ HWID saved to hwid.txt")
        sys.exit(0)

    if not args.config:
        parser.error("the following argument is required: --config")

    validate_license(args.license)

    detector = MultiCamDetector(model_path=args.model, confidence_threshold=args.conf)
    detector.run_from_config(config_file=args.config)
