import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import math

from PySide6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap


# ================= SETTINGS ================= #

MODEL_PATH = "D:/Python/Media Pipe/hand_landmarker.task"
FACE_MODEL_PATH = "D:/Python/Media Pipe/face_landmarker.task"

SMOOTHING = 10
PINCH_THRESHOLD_RATIO = 0.4
PINCH_REQUIRED = 3
FIST_REQUIRED = 3
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

FACE_KEY_POINTS = {
    "nose_tip": 1,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

FACE_CONTOURS = [
    # right eye
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    # left eye
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),(382,362),
    (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362),
    # right eyebrow
    (46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107),
    # left eyebrow
    (276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336),
    # lips outer
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(61,185),(185,40),(40,39),(39,37),(37,0),(0,267),
    (267,269),(269,270),(270,409),(409,291),
    # face oval
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),
    (454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),
    (400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),
    (54,103),(103,67),(67,109),(109,10),
]

# ================= CAMERA / GESTURE / FACE THREAD ================= #

class GestureThread(QThread):
    frame_ready = Signal(np.ndarray)
    status_updated = Signal(str, int, float)
    mode_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.mode = "hand"

    def set_mode(self, mode):
        self.mode = mode
        self.mode_changed.emit(mode)

    def run(self):
        hand_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options, num_hands=1
        )
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        face_detector = None   # lazy-loaded on first switch to face mode


        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       
        

        prev_x = 0
        prev_y = 0
        holding = False
        pinch_frames = 0
        fist_frames = 0
        fist_active = False
        scroll_mode = False
        palm_hold_start_time = None
        scroll_neutral_y = None
        non_palm_time = None
        previous_time = time.time()

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

            # ============ FACE MODE ============ #
            if self.mode == "face":

                if face_detector is None:
                    face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
                    face_options = vision.FaceLandmarkerOptions(
                        base_options=face_base_options, num_faces=1
                    )
                    face_detector = vision.FaceLandmarker.create_from_options(face_options)

                face_results = face_detector.detect(mp_image)

                if face_results.face_landmarks:
                    face = face_results.face_landmarks[0]

                    xs = [pt.x * w for pt in face]
                    ys = [pt.y * h for pt in face]
                    min_x, max_x = int(min(xs)), int(max(xs))
                    min_y, max_y = int(min(ys)), int(max(ys))

                    pad = 15
                    min_x -= pad; min_y -= pad
                    max_x += pad; max_y += pad

                    box_w = max_x - min_x
                    box_h = max_y - min_y
                    bracket_len = int(min(box_w, box_h) * 0.22)
                    tick_len = int(bracket_len * 0.35)
                    color = (255, 220, 0)

                    pulse = 0.5 + 0.5 * math.sin(now * 4)
                    thin = 2
                    thick_glow = 6

                    glow_layer = frame.copy()

                    corners = [
                        (min_x, min_y, 1, 1),
                        (max_x, min_y, -1, 1),
                        (min_x, max_y, 1, -1),
                        (max_x, max_y, -1, -1),
                    ]
                    for cx, cy, sign_x, sign_y in corners:
                        cv2.line(glow_layer, (cx, cy), (cx + sign_x*bracket_len, cy), color, thick_glow)
                        cv2.line(glow_layer, (cx, cy), (cx, cy + sign_y*bracket_len), color, thick_glow)
                        cv2.line(glow_layer, (cx + sign_x*bracket_len, cy),
                                 (cx + sign_x*(bracket_len+tick_len), cy), color, thick_glow)
                        cv2.line(glow_layer, (cx, cy + sign_y*bracket_len),
                                 (cx, cy + sign_y*(bracket_len+tick_len)), color, thick_glow)

                    glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)
                    alpha = 0.35 + 0.25 * pulse
                    frame[:] = cv2.addWeighted(glow_layer, alpha, frame, 1 - alpha, 0)

                    for cx, cy, sign_x, sign_y in corners:
                        cv2.line(frame, (cx, cy), (cx + sign_x*bracket_len, cy), color, thin)
                        cv2.line(frame, (cx, cy), (cx, cy + sign_y*bracket_len), color, thin)
                        cv2.line(frame, (cx + sign_x*bracket_len, cy),
                                 (cx + sign_x*(bracket_len+tick_len), cy), color, thin)
                        cv2.line(frame, (cx, cy + sign_y*bracket_len),
                                 (cx, cy + sign_y*(bracket_len+tick_len)), color, thin)

                    scan_progress = (now * 0.6) % 1.0
                    scan_x = int(min_x + scan_progress * box_w)
                    cv2.line(frame, (scan_x, min_y), (scan_x, max_y), (0, 255, 255), 1)

                    ncx = (min_x + max_x) // 2
                    ncy = (min_y + max_y) // 2
                    gap = 6
                    arm = 14
                    cross_color = (0, 255, 255)
                    cv2.line(frame, (ncx - arm, ncy), (ncx - gap, ncy), cross_color, 1)
                    cv2.line(frame, (ncx + gap, ncy), (ncx + arm, ncy), cross_color, 1)
                    cv2.line(frame, (ncx, ncy - arm), (ncx, ncy - gap), cross_color, 1)
                    cv2.line(frame, (ncx, ncy + arm), (ncx, ncy + gap), cross_color, 1)
                    cv2.circle(frame, (ncx, ncy), 2, cross_color, -1)

                    # subtle wireframe contour overlay (eyes, brows, lips, jawline)
                    wireframe_layer = frame.copy()
                    for start_idx, end_idx in FACE_CONTOURS:
                        p1 = face[start_idx]
                        p2 = face[end_idx]
                        x1, y1 = int(p1.x * w), int(p1.y * h)
                        x2, y2 = int(p2.x * w), int(p2.y * h)
                        cv2.line(wireframe_layer, (x1, y1), (x2, y2), (255, 220, 0), 1)

                    frame[:] = cv2.addWeighted(wireframe_layer, 0.45, frame, 0.55, 0)

                    for name, idx in FACE_KEY_POINTS.items():
                        pt = face[idx]
                        px, py = int(pt.x * w), int(pt.y * h)
                        cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

                    cv2.putText(frame, "FACE LOCKED", (min_x, min_y - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    status = "FACE LOCKED"
                else:
                    status = "NO FACE DETECTED"

            # ============ HAND MODE ============ #
            else:

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

                        elif fingers == 0:
                            palm_hold_start_time = None
                            fist_frames += 1

                            target_x = np.interp(index.x, [0.15, 0.85], [0, SCREEN_W])
                            target_y = np.interp(index.y, [0.15, 0.85], [0, SCREEN_H])
                            prev_x = prev_x + (target_x - prev_x) / SMOOTHING
                            prev_y = prev_y + (target_y - prev_y) / SMOOTHING

                            if fist_frames >= FIST_REQUIRED and not fist_active:
                                pyautogui.rightClick()
                                fist_active = True

                            status = "FIST (Right-Click)"

                        else:
                            palm_hold_start_time = None
                            fist_frames = 0
                            fist_active = False

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
                    scroll_mode = False
                    scroll_neutral_y = None
                    palm_hold_start_time = None
                    non_palm_time = None
                    if holding:
                        pyautogui.mouseUp()
                        holding = False
                    status = "NO HAND DETECTED"

            previous_time = now

            self.frame_ready.emit(frame)
            self.status_updated.emit(status, fingers, normalized_distance)

        cap.release()

    def stop(self):
        self.running = False
        
      


# ================= HUD WINDOW ================= #

class TitanHUD(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)

        self.drag_position = QPoint()

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(10, 10, 10, 10)

        title_row = QHBoxLayout()
        title_label = QLabel("TITAN HUD")
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

        self.mode_label = QLabel("MODE: HAND GESTURE   (Press F = Face, H = Hand)")
        self.mode_label.setStyleSheet("color: #00ffcc; font-size: 14px; font-weight: bold;")

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(480, 360)
        self.camera_label.setStyleSheet("background-color: black; border-radius: 8px;")

        self.status_label = QLabel("Status: Starting...")
        self.fingers_label = QLabel("Fingers: -")
        self.pinch_label = QLabel("Pinch Ratio: -")

        for lbl in (self.status_label, self.fingers_label, self.pinch_label):
            lbl.setStyleSheet("color: white; font-size: 16px;")

        outer_layout.addLayout(title_row)
        outer_layout.addWidget(self.mode_label)
        outer_layout.addWidget(self.camera_label)
        outer_layout.addWidget(self.status_label)
        outer_layout.addWidget(self.fingers_label)
        outer_layout.addWidget(self.pinch_label)

        self.setLayout(outer_layout)
        self.setStyleSheet("""
            TitanHUD {
                background-color: rgba(15, 15, 20, 200);
                border-radius: 14px;
            }
        """)
        self.setGeometry(80, 80, 520, 510)

        self.gesture_thread = GestureThread()
        self.gesture_thread.frame_ready.connect(self.update_frame)
        self.gesture_thread.status_updated.connect(self.update_status)
        self.gesture_thread.mode_changed.connect(self.update_mode_label)
        self.gesture_thread.start()

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

    def update_mode_label(self, mode):
        if mode == "face":
            self.mode_label.setText("MODE: FACE TRACKING   (Press F = Face, H = Hand)")
        else:
            self.mode_label.setText("MODE: HAND GESTURE   (Press F = Face, H = Hand)")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.gesture_thread.set_mode("face")
        elif event.key() == Qt.Key_H:
            self.gesture_thread.set_mode("hand")

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
        self.gesture_thread.wait()
        event.accept()


app = QApplication(sys.argv)
hud = TitanHUD()
hud.show()
sys.exit(app.exec())