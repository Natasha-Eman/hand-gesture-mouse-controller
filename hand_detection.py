import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup
base_options = python.BaseOptions(
    model_asset_path='D:/Media Pipe/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Convert frame for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, 
        data=rgb_frame)
    
    # Detect hands
    results = detector.detect(mp_image)
    
    # Draw landmarks if hand detected
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            for landmark in hand:
                h, w, _ = frame.shape
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    
    cv2.imshow("Hand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()