import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import math


# ================= SETTINGS ================= #

MODEL_PATH = "D:/Python/Media Pipe/hand_landmarker.task"

SMOOTHING = 10

PINCH_THRESHOLD_RATIO = 0.4
PINCH_REQUIRED = 3

FIST_REQUIRED = 3

GESTURE_HOLD_MS = 300      # time to hold open palm before entering scroll mode
EXIT_HOLD_MS = 150         # time to hold non-palm gesture before exiting scroll mode
DEAD_ZONE = 0.03           # ignore small offsets near neutral position
SCROLL_SENSITIVITY = 1000

pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()



# ================= FINGER COUNTER ================= #

def count_fingers(hand):

    fingers = []
    wrist = hand[0].x

    if abs(hand[4].x - wrist) > abs(hand[3].x - wrist):
        fingers.append(1)
    else:
        fingers.append(0)

    tips = [8,12,16,20]

    for tip in tips:
        if hand[tip].y < hand[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)




# ================= HAND CONNECTIONS ================= #

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]



# ================= MEDIAPIPE ================= #

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)




# ================= CAMERA ================= #

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)



# ================= STATE VARIABLES ================= #

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


cv2.namedWindow("AI Hand Gesture Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Hand Gesture Mouse", 640, 480)



# ================= MAIN LOOP ================= #

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape

    status = "RELEASED"
    status_color = (0,255,0)
    normalized_distance = 0.0
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    cv2.putText(frame, "AI Hand Gesture Virtual Mouse", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(frame, "Press Q to Quit", (450,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        for point in hand:
            x = int(point.x*w)
            y = int(point.y*h)
            cv2.circle(frame, (x,y), 7, (255,0,255), -1)

        for start,end in connections:
            x1=int(hand[start].x*w); y1=int(hand[start].y*h)
            x2=int(hand[end].x*w); y2=int(hand[end].y*h)
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 3)

        index = hand[8]
        thumb = hand[4]
        wrist_point = hand[0]

        ix=int(index.x*w); iy=int(index.y*h)
        tx=int(thumb.x*w); ty=int(thumb.y*h)

        fingers = count_fingers(hand)


        # ================= SCROLL MODE HANDLING ================= #

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

                # keep cursor tracking in sync during scroll mode too
                target_x=np.interp(index.x, [0.15,0.85], [0,SCREEN_W])
                target_y=np.interp(index.y, [0.15,0.85], [0,SCREEN_H])
                prev_x = prev_x + (target_x-prev_x)/SMOOTHING
                prev_y = prev_y + (target_y-prev_y)/SMOOTHING
                offset = wrist_point.y - scroll_neutral_y

                if abs(offset) > DEAD_ZONE:
                    sign = -1 if offset > 0 else 1
                    scroll_amount = int(sign * (offset ** 2) * SCROLL_SENSITIVITY)

                    MAX_SCROLL_SPEED = 40
                    scroll_amount = max(-MAX_SCROLL_SPEED, min(MAX_SCROLL_SPEED, scroll_amount))

                    pyautogui.scroll(scroll_amount)

                status = "SCROLL MODE"
                status_color = (0,255,255)

            fist_frames = 0
            fist_active = False


        # ================= NOT IN SCROLL MODE ================= #

        else:

            if fingers == 5:
                # ---- holding open palm, checking entry debounce ----
                if palm_hold_start_time is None:
                    palm_hold_start_time = now

                elif (now - palm_hold_start_time) * 1000 >= GESTURE_HOLD_MS:
                    scroll_mode = True
                    scroll_neutral_y = wrist_point.y
                    non_palm_time = None

                status = "HOLD FOR SCROLL MODE..."
                status_color = (0,200,200)

                fist_frames = 0
                fist_active = False


            elif fingers == 0:
                # ---- FIST: right-click, once per fist ----
                palm_hold_start_time = None
                fist_frames += 1

                # keep cursor tracking in sync so it doesn't snap when returning to normal
                target_x=np.interp(index.x, [0.15,0.85], [0,SCREEN_W])
                target_y=np.interp(index.y, [0.15,0.85], [0,SCREEN_H])
                prev_x = prev_x + (target_x-prev_x)/SMOOTHING
                prev_y = prev_y + (target_y-prev_y)/SMOOTHING

                if fist_frames >= FIST_REQUIRED and not fist_active:
                    pyautogui.rightClick()
                    fist_active = True

                status = "FIST (Right-Click)"
                status_color = (255,0,0)


            else:
                # ---- NORMAL: cursor movement + pinch click/drag ----
                palm_hold_start_time = None
                fist_frames = 0
                fist_active = False

                target_x=np.interp(index.x, [0.15,0.85], [0,SCREEN_W])
                target_y=np.interp(index.y, [0.15,0.85], [0,SCREEN_H])

                current_x = prev_x + (target_x-prev_x)/SMOOTHING
                current_y = prev_y + (target_y-prev_y)/SMOOTHING

                pyautogui.moveTo(current_x, current_y)

                prev_x=current_x
                prev_y=current_y

                distance = math.sqrt((ix-tx)**2 + (iy-ty)**2)

                middle_mcp = hand[9]
                hand_scale = math.sqrt(
                    (wrist_point.x*w - middle_mcp.x*w)**2 +
                    (wrist_point.y*h - middle_mcp.y*h)**2
                )
                if hand_scale == 0:
                    hand_scale = 1

                normalized_distance = distance / hand_scale

                if normalized_distance < PINCH_THRESHOLD_RATIO:
                    pinch_frames += 1
                    if pinch_frames >= PINCH_REQUIRED:
                        if not holding:
                            pyautogui.mouseDown()
                            holding=True
                        status="HOLDING"
                        status_color=(0,0,255)
                else:
                    pinch_frames=0
                    if holding:
                        pyautogui.mouseUp()
                        holding=False
                    status="RELEASED"
                    status_color=(0,255,0)


        # ================= TEXT ================= #

        cv2.putText(frame, status, (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Pinch Ratio: {normalized_distance:.2f}", (30,150),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
        cv2.putText(frame, f"Fingers: {fingers}", (30,190),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)


    else:

        pinch_frames=0
        fist_frames = 0
        fist_active = False
        scroll_mode = False
        scroll_neutral_y = None
        palm_hold_start_time = None
        non_palm_time = None

        if holding:
            pyautogui.mouseUp()
            holding=False

        cv2.putText(frame, "NO HAND DETECTED", (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    # ================= FPS ================= #

    fps = 1/(now-previous_time)
    previous_time=now
    cv2.putText(frame, f"FPS: {int(fps)}", (30,240),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.putText(frame, f"Sensitivity: {SCROLL_SENSITIVITY}", (30,280),
            cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)

    # ================= DISPLAY ================= #

    small = cv2.resize(frame, (640,480))
    cv2.imshow("AI Hand Gesture Mouse", small)
    cv2.setWindowProperty("AI Hand Gesture Mouse", cv2.WND_PROP_TOPMOST, 1)

    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        SCROLL_SENSITIVITY += 50
    elif key == ord('-'):
        SCROLL_SENSITIVITY = max(0, SCROLL_SENSITIVITY - 50)
 
cap.release()
cv2.destroyAllWindows()