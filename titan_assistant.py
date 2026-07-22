import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import math
import pyaudio
import wave
import winsound
import subprocess
import psutil
import pywhatkit

from PySide6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap
from openwakeword.model import Model as OWWModel
from faster_whisper import WhisperModel


# ================= SETTINGS ================= #

MODEL_PATH = "D:/Python/Media Pipe/hand_landmarker.task"

SMOOTHING = 10
PINCH_THRESHOLD_RATIO = 0.4
PINCH_REQUIRED = 3
FIST_REQUIRED = 3
DOUBLE_CLICK_REQUIRED = 3
GESTURE_HOLD_MS = 300
EXIT_HOLD_MS = 150
DEAD_ZONE = 0.03
SCROLL_SENSITIVITY = 1000

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()


def count_fingers(hand):
    fingers = []
    wrist = hand[0].x
    if abs(hand[4].x - wrist) > abs(hand[3].x - wrist):
        fingers.append(1)
    else:
        fingers.append(0)
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand[tip].y < hand[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ================= GESTURE THREAD ================= #

class GestureThread(QThread):
    frame_ready = Signal(np.ndarray)
    status_updated = Signal(str, int, float)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        try:
            hand_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base_options, num_hands=1
            )
            hand_detector = vision.HandLandmarker.create_from_options(hand_options)

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            prev_x = 0
            prev_y = 0
            holding = False
            pinch_frames = 0
            fist_frames = 0
            fist_active = False
            double_click_frames = 0
            double_click_active = False
            scroll_mode = False
            palm_hold_start_time = None
            scroll_neutral_y = None
            non_palm_time = None

            while self.running:

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                status = "RELEASED"
                normalized_distance = 0.0
                fingers = 0
                now = time.time()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                results = hand_detector.detect(mp_image)

                if results.hand_landmarks:

                    hand = results.hand_landmarks[0]

                    for point in hand:
                        x = int(point.x * w); y = int(point.y * h)
                        cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)

                    for start, end in HAND_CONNECTIONS:
                        x1 = int(hand[start].x * w); y1 = int(hand[start].y * h)
                        x2 = int(hand[end].x * w); y2 = int(hand[end].y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    index = hand[8]
                    thumb = hand[4]
                    wrist_point = hand[0]

                    ix = int(index.x * w); iy = int(index.y * h)
                    tx = int(thumb.x * w); ty = int(thumb.y * h)

                    fingers = count_fingers(hand)

                    if scroll_mode:
                        if fingers != 5:
                            if non_palm_time is None:
                                non_palm_time = now
                            elif (now - non_palm_time) * 1000 >= EXIT_HOLD_MS:
                                scroll_mode = False
                                scroll_neutral_y = None
                                non_palm_time = None
                        else:
                            non_palm_time = None

                        if scroll_mode:
                            target_x = np.interp(index.x, [0.15, 0.85], [0, SCREEN_W])
                            target_y = np.interp(index.y, [0.15, 0.85], [0, SCREEN_H])
                            prev_x = prev_x + (target_x - prev_x) / SMOOTHING
                            prev_y = prev_y + (target_y - prev_y) / SMOOTHING

                            offset = wrist_point.y - scroll_neutral_y
                            if abs(offset) > DEAD_ZONE:
                                sign = -1 if offset > 0 else 1
                                scroll_amount = int(sign * (offset ** 2) * SCROLL_SENSITIVITY)
                                MAX_SCROLL_SPEED = 40
                                scroll_amount = max(-MAX_SCROLL_SPEED, min(MAX_SCROLL_SPEED, scroll_amount))
                                pyautogui.scroll(scroll_amount)

                            status = "SCROLL MODE"

                        fist_frames = 0
                        fist_active = False
                        double_click_frames = 0
                        double_click_active = False

                    else:
                        if fingers == 5:
                            if palm_hold_start_time is None:
                                palm_hold_start_time = now
                            elif (now - palm_hold_start_time) * 1000 >= GESTURE_HOLD_MS:
                                scroll_mode = True
                                scroll_neutral_y = wrist_point.y
                                non_palm_time = None

                            status = "HOLD FOR SCROLL..."
                            fist_frames = 0
                            fist_active = False
                            double_click_frames = 0
                            double_click_active = False

                        elif fingers == 0:
                            palm_hold_start_time = None
                            fist_frames += 1
                            double_click_frames = 0
                            double_click_active = False

                            target_x = np.interp(index.x, [0.15, 0.85], [0, SCREEN_W])
                            target_y = np.interp(index.y, [0.15, 0.85], [0, SCREEN_H])
                            prev_x = prev_x + (target_x - prev_x) / SMOOTHING
                            prev_y = prev_y + (target_y - prev_y) / SMOOTHING

                            if fist_frames >= FIST_REQUIRED and not fist_active:
                                pyautogui.rightClick()
                                fist_active = True

                            status = "FIST (Right-Click)"

                        elif fingers == 2:
                            palm_hold_start_time = None
                            fist_frames = 0
                            fist_active = False
                            double_click_frames += 1

                            target_x = np.interp(index.x, [0.15, 0.85], [0, SCREEN_W])
                            target_y = np.interp(index.y, [0.15, 0.85], [0, SCREEN_H])
                            prev_x = prev_x + (target_x - prev_x) / SMOOTHING
                            prev_y = prev_y + (target_y - prev_y) / SMOOTHING

                            if double_click_frames >= DOUBLE_CLICK_REQUIRED and not double_click_active:
                                pyautogui.doubleClick()
                                double_click_active = True

                            status = "TWO FINGERS (Double-Click)"

                        else:
                            palm_hold_start_time = None
                            fist_frames = 0
                            fist_active = False
                            double_click_frames = 0
                            double_click_active = False

                            target_x = np.interp(index.x, [0.15, 0.85], [0, SCREEN_W])
                            target_y = np.interp(index.y, [0.15, 0.85], [0, SCREEN_H])
                            current_x = prev_x + (target_x - prev_x) / SMOOTHING
                            current_y = prev_y + (target_y - prev_y) / SMOOTHING
                            pyautogui.moveTo(current_x, current_y)
                            prev_x = current_x
                            prev_y = current_y

                            distance = math.sqrt((ix - tx) ** 2 + (iy - ty) ** 2)
                            middle_mcp = hand[9]
                            hand_scale = math.sqrt(
                                (wrist_point.x * w - middle_mcp.x * w) ** 2 +
                                (wrist_point.y * h - middle_mcp.y * h) ** 2
                            )
                            if hand_scale == 0:
                                hand_scale = 1
                            normalized_distance = distance / hand_scale

                            if normalized_distance < PINCH_THRESHOLD_RATIO:
                                pinch_frames += 1
                                if pinch_frames >= PINCH_REQUIRED:
                                    if not holding:
                                        pyautogui.mouseDown()
                                        holding = True
                                    status = "HOLDING"
                            else:
                                pinch_frames = 0
                                if holding:
                                    pyautogui.mouseUp()
                                    holding = False
                                status = "RELEASED"

                else:
                    pinch_frames = 0
                    fist_frames = 0
                    fist_active = False
                    double_click_frames = 0
                    double_click_active = False
                    scroll_mode = False
                    scroll_neutral_y = None
                    palm_hold_start_time = None
                    non_palm_time = None
                    if holding:
                        pyautogui.mouseUp()
                        holding = False
                    status = "NO HAND DETECTED"

                self.frame_ready.emit(frame)
                self.status_updated.emit(status, fingers, normalized_distance)

            cap.release()

        except Exception as e:
            self.status_updated.emit(f"ERROR: {e}", 0, 0.0)

    def stop(self):
        self.running = False


# ================= VOICE THREAD ================= #

class VoiceThread(QThread):
    voice_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def handle_command(self, text):
        text = text.lower()

        if "battery" in text:
            battery = psutil.sensors_battery()
            if battery:
                msg = f"Battery is at {int(battery.percent)}%"
            else:
                msg = "Battery info not available"

        elif "volume up" in text:
            for _ in range(5):
                pyautogui.press("volumeup")
            msg = "Volume increased"

        elif "volume down" in text:
            for _ in range(5):
                pyautogui.press("volumedown")
            msg = "Volume decreased"

        elif "mute" in text:
            pyautogui.press("volumemute")
            msg = "Muted"

        elif text.startswith("play "):
            query = text.replace("play ", "", 1).replace(" on youtube", "").strip()
            msg = f"Playing '{query}' on YouTube"
            try:
                pywhatkit.playonyt(query)
            except Exception as e:
                msg = f"Could not play video: {e}"

        elif "search" in text and "google" in text:
            query = text.replace("search", "").replace("on google", "").replace("google", "").strip()
            msg = f"Searching Google for '{query}'"
            try:
                pywhatkit.search(query)
            except Exception as e:
                msg = f"Could not search: {e}"

        elif "chrome" in text:
            subprocess.Popen(["start", "chrome"], shell=True)
            msg = "Opening Chrome"

        elif "notepad" in text:
            subprocess.Popen(["notepad.exe"])
            msg = "Opening Notepad"

        elif "scroll up" in text:
            pyautogui.scroll(300)
            msg = "Scrolled up"

        elif "scroll down" in text:
            pyautogui.scroll(-300)
            msg = "Scrolled down"

        else:
            msg = f"No matching command for: \"{text}\""

        self.voice_status.emit(f"Heard: \"{text}\" → {msg}")

    def run(self):
        try:
            owwModel = OWWModel(wakeword_models=["hey_jarvis"])
            whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            CHUNK = 1280
            RECORD_SECONDS = 4
            COOLDOWN_SECONDS = 2

            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                 input=True, frames_per_buffer=CHUNK)

            self.voice_status.emit("Listening for 'Hey Jarvis'...")

            last_trigger_time = 0
            armed = True

            while self.running:
                audio_chunk = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                prediction = owwModel.predict(audio_chunk)

                now = time.time()

                for wakeword, score in prediction.items():
                    if score > 0.5 and armed and (now - last_trigger_time) > COOLDOWN_SECONDS:
                        armed = False

                        winsound.Beep(1000, 150)
                        self.voice_status.emit("Listening for command...")

                        frames = []
                        num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
                        for _ in range(num_chunks):
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            frames.append(data)

                        wf = wave.open("command.wav", "wb")
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b"".join(frames))
                        wf.close()

                        segments, _ = whisper_model.transcribe("command.wav")
                        text = " ".join([seg.text for seg in segments]).strip()

                        self.handle_command(text)
                        winsound.Beep(1500, 100)

                        last_trigger_time = time.time()

                    elif score < 0.2:
                        armed = True

            stream.stop_stream()
            stream.close()
            audio.terminate()

        except Exception as e:
            self.voice_status.emit(f"ERROR: {e}")

    def stop(self):
        self.running = False


# ================= HUD WINDOW ================= #

class TitanAssistant(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)

        self.drag_position = QPoint()

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(10, 10, 10, 10)

        title_row = QHBoxLayout()
        title_label = QLabel("TITAN ASSISTANT")
        title_label.setStyleSheet("color: cyan; font-size: 16px; font-weight: bold;")
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton { color: white; background-color: rgba(255,0,0,150); border-radius: 12px; }
            QPushButton:hover { background-color: rgba(255,0,0,220); }
        """)
        close_btn.clicked.connect(self.close)
        title_row.addWidget(title_label)
        title_row.addStretch()
        title_row.addWidget(close_btn)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(480, 360)
        self.camera_label.setStyleSheet("background-color: black; border-radius: 8px;")

        self.status_label = QLabel("Status: Starting...")
        self.fingers_label = QLabel("Fingers: -")
        self.pinch_label = QLabel("Pinch Ratio: -")
        self.voice_label = QLabel("Voice: Starting...")
        self.voice_label.setStyleSheet("color: #ffaa00; font-size: 14px;")
        self.voice_label.setWordWrap(True)

        for lbl in (self.status_label, self.fingers_label, self.pinch_label):
            lbl.setStyleSheet("color: white; font-size: 16px;")

        outer_layout.addLayout(title_row)
        outer_layout.addWidget(self.camera_label)
        outer_layout.addWidget(self.status_label)
        outer_layout.addWidget(self.fingers_label)
        outer_layout.addWidget(self.pinch_label)
        outer_layout.addWidget(self.voice_label)

        self.setLayout(outer_layout)
        self.setStyleSheet("""
            TitanAssistant {
                background-color: rgba(15, 15, 20, 200);
                border-radius: 14px;
            }
        """)
        self.setGeometry(80, 80, 520, 480)

        self.gesture_thread = GestureThread()
        self.gesture_thread.frame_ready.connect(self.update_frame)
        self.gesture_thread.status_updated.connect(self.update_status)
        self.gesture_thread.start()

        self.voice_thread = VoiceThread()
        self.voice_thread.voice_status.connect(self.update_voice_status)
        self.voice_thread.start()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio
        )
        self.camera_label.setPixmap(pixmap)

    def update_status(self, status, fingers, pinch_ratio):
        self.status_label.setText(f"Status: {status}")
        self.fingers_label.setText(f"Fingers: {fingers}")
        self.pinch_label.setText(f"Pinch Ratio: {pinch_ratio:.2f}")

    def update_voice_status(self, text):
        self.voice_label.setText(f"Voice: {text}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        self.setFocus()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def closeEvent(self, event):
        self.gesture_thread.stop()
        self.gesture_thread.wait(3000)
        self.voice_thread.stop()
        self.voice_thread.wait(2000)
        event.accept()


app = QApplication(sys.argv)
assistant = TitanAssistant()
assistant.show()
sys.exit(app.exec())