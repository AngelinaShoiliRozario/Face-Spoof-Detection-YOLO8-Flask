import math
import time
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6

cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("./models/my2.mp4")  # For Video


model = YOLO("./models/best.pt")

classNames = ["real", "fake"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split the HSV image into its components
    h, s, v = cv2.split(hsv)

    # Adjust the value (brightness) channel
    v = cv2.multiply(v, 1.2)

    # Clip the values to the range [0, 255]
    v = np.clip(v, 0, 255)

    # Merge the HSV components back together
    hsv = cv2.merge([h, s, v])

    # Convert the HSV image back to the BGR color space
    brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    results = model(brightened_frame, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print('confidence ',conf)
            # Class Name
            cls = int(box.cls[0])
            print('class ',cls)
            if conf > confidence:

                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(brightened_frame, (x1, y1, w, h),colorC=color,colorR=color)
                cvzone.putTextRect(brightened_frame, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
                                   colorB=color)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", brightened_frame)
    cv2.waitKey(1)
