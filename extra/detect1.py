import math
import time

import cv2
import cvzone
from ultralytics import YOLO


def detect_spoof(camera):

    confidence = 0.8

    cap = camera  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)
    # cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

    model = YOLO("./models/best.pt")

    classNames = ["real", "fake"]

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True,verbose=False)
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

                    ret, buffer = cv2.imencode('.jpg', img) # image to buffer
                    frame = buffer.tobytes()


                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


