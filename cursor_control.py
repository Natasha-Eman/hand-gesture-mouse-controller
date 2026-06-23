import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import numpy as np


# Screen size
screen_w, screen_h = pyautogui.size()

# PyAutoGUI safety
pyautogui.FAILSAFE = False


# Setup MediaPipe
base_options = python.BaseOptions(
    model_asset_path='D:/Media Pipe/hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# Open camera
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    h, w, _ = frame.shape


    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)


    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hands
    results = detector.detect(mp_image)


    if results.hand_landmarks:

        for hand in results.hand_landmarks:


            # Index finger tip
            index_tip = hand[8]

            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)


            # Thumb tip
            thumb_tip = hand[4]

            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)



            # Draw index circle
            cv2.circle(
                frame,
                (cx, cy),
                15,
                (255, 0, 0),
                -1
            )


            # Draw thumb circle
            cv2.circle(
                frame,
                (tx, ty),
                15,
                (0, 255, 0),
                -1
            )



            # Better mouse mapping
            screen_x = int(
                np.interp(
                    index_tip.x,
                    [0.2, 0.8],
                    [0, screen_w]
                )
            )

            screen_y = int(
                np.interp(
                    index_tip.y,
                    [0.1, 0.9],
                    [0, screen_h]
                )
            )


            # Move cursor
            pyautogui.moveTo(
                screen_x,
                screen_y
            )



            # Distance between thumb and index
            distance = (
                (cx - tx)**2 +
                (cy - ty)**2
            )**0.5



            # Pinch = HOLD
            if distance < 40:

                pyautogui.mouseDown()

                cv2.putText(
                    frame,
                    "HOLDING!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )


            else:

                pyautogui.mouseUp()

                cv2.putText(
                    frame,
                    "RELEASED!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )



    cv2.imshow(
        "Cursor Control",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release mouse if program closes
pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()