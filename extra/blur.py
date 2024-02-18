import cv2
import cv2
import numpy as np
from ultralytics import YOLO

# Open webcam
cap = cv2.VideoCapture(1)  # For Webcam
model = YOLO("./models/best.pt")
# Set desired resolution
cap.set(3, 640)
cap.set(4, 480)

# Brightness adjustment factor (increase to brighten the image)
brightness_factor = 1.2

def compute_blurriness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Compute the variance of the Laplacian
    variance = laplacian.var()
    return variance



while True:

    success, img = cap.read()

    results = model(img, stream=True,verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Crop the detected face region from the original image
            face_roi = img[y1:y2, x1:x2]

            # Check the blurriness of the image
            blurriness = compute_blurriness(face_roi)

            print("Blurriness:", blurriness)
            cv2.imshow("Image", face_roi)
            cv2.waitKey(1)