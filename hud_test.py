import cv2
import time

print(f"[{time.time():.2f}] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"[{time.time():.2f}] Camera opened.")
cap.release()