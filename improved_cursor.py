import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import math


# ================= SETTINGS ================= #

MODEL_PATH = "D:/Media Pipe/hand_landmarker.task"

SMOOTHING = 10

PINCH_THRESHOLD = 35
PINCH_REQUIRED = 3

pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()



# ================= FINGER COUNTER ================= #

def count_fingers(hand):

    fingers = []

    # Thumb
    wrist = hand[0].x

    if abs(hand[4].x - wrist) > abs(hand[3].x - wrist):
        fingers.append(1)
    else:
        fingers.append(0)



    # Other fingers

    tips = [8,12,16,20]

    for tip in tips:

        if hand[tip].y < hand[tip-2].y:
            fingers.append(1)

        else:
            fingers.append(0)


    return fingers.count(1)




# ================= HAND CONNECTIONS ================= #

connections = [

    # thumb
    (0,1),(1,2),(2,3),(3,4),

    # index
    (0,5),(5,6),(6,7),(7,8),

    # middle
    (0,9),(9,10),(10,11),(11,12),

    # ring
    (0,13),(13,14),(14,15),(15,16),

    # pinky
    (0,17),(17,18),(18,19),(19,20),

    # palm
    (5,9),(9,13),(13,17)

]



# ================= MEDIAPIPE ================= #

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)


detector = vision.HandLandmarker.create_from_options(options)




# ================= CAMERA ================= #

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)



# Cursor variables

prev_x = 0
prev_y = 0


holding = False


pinch_frames = 0


previous_time = time.time()



# Window setup

cv2.namedWindow(
    "AI Hand Gesture Mouse",
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    "AI Hand Gesture Mouse",
    640,
    480
)



# ================= MAIN LOOP ================= #

while True:


    ret, frame = cap.read()


    if not ret:
        break



    # mirror effect

    frame = cv2.flip(frame,1)


    h,w,_ = frame.shape



    status = "RELEASED"
    status_color = (0,255,0)



    # Convert image

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    results = detector.detect(mp_image)




    # ================= HEADER ================= #

    cv2.putText(
        frame,
        "AI Hand Gesture Virtual Mouse",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to Quit",
        (450,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )




    if results.hand_landmarks:


        hand = results.hand_landmarks[0]



        # ================= DRAW SKELETON ================= #


        # draw joints

        for point in hand:

            x = int(point.x*w)
            y = int(point.y*h)


            cv2.circle(
                frame,
                (x,y),
                7,
                (255,0,255),
                -1
            )



        # draw bones

        for start,end in connections:


            x1=int(hand[start].x*w)
            y1=int(hand[start].y*h)


            x2=int(hand[end].x*w)
            y2=int(hand[end].y*h)



            cv2.line(
                frame,
                (x1,y1),
                (x2,y2),
                (255,255,255),
                3
            )




        # points

        index = hand[8]
        thumb = hand[4]



        ix=int(index.x*w)
        iy=int(index.y*h)


        tx=int(thumb.x*w)
        ty=int(thumb.y*h)




        # ================= CURSOR ================= #


        target_x=np.interp(
            index.x,
            [0.15,0.85],
            [0,SCREEN_W]
        )


        target_y=np.interp(
            index.y,
            [0.15,0.85],
            [0,SCREEN_H]
        )



        current_x = (
            prev_x +
            (target_x-prev_x)/SMOOTHING
        )


        current_y = (
            prev_y +
            (target_y-prev_y)/SMOOTHING
        )



        pyautogui.moveTo(
            current_x,
            current_y
        )



        prev_x=current_x
        prev_y=current_y




        # ================= PINCH ================= #


        distance = math.sqrt(
            (ix-tx)**2 +
            (iy-ty)**2
        )



        fingers = count_fingers(hand)



        if distance < PINCH_THRESHOLD:


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


        cv2.putText(
            frame,
            status,
            (30,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )



        cv2.putText(
            frame,
            f"Pinch Distance: {int(distance)}",
            (30,150),
            cv2.FONT_HERSHEY_SIMPLEX,
            .8,
            (255,255,255),
            2
        )


        cv2.putText(
            frame,
            f"Fingers: {fingers}",
            (30,190),
            cv2.FONT_HERSHEY_SIMPLEX,
            .8,
            (255,255,255),
            2
        )




    else:


        pinch_frames=0


        if holding:

            pyautogui.mouseUp()
            holding=False



        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (30,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )




    # ================= FPS ================= #

    now=time.time()

    fps = 1/(now-previous_time)

    previous_time=now



    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30,240),
        cv2.FONT_HERSHEY_SIMPLEX,
        .8,
        (255,255,255),
        2
    )




    # ================= DISPLAY ================= #

    small = cv2.resize(
        frame,
        (640,480)
    )


    cv2.imshow(
        "AI Hand Gesture Mouse",
        small
    )


    cv2.setWindowProperty(
        "AI Hand Gesture Mouse",
        cv2.WND_PROP_TOPMOST,
        1
    )



    if cv2.waitKey(1)&0xFF == ord('q'):
        break




# cleanup

pyautogui.mouseUp()

cap.release()

cv2.destroyAllWindows()