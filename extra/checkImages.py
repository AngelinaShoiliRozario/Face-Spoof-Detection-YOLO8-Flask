import math
import time
import os
import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 640)
# cap.set(4, 480)
# cap = cv2.VideoCapture("./models/my2.mp4")  # For Video
brightness_factor = 1.2

def compute_blurriness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Compute the variance of the Laplacian
    variance = laplacian.var()
    return variance
def check_reflection(image):
    pass

model = YOLO("../models/best7000.pt")

classNames = ["real", "fake"]

prev_frame_time = 0
new_frame_time = 0

folder_path = "datasets"
files = os.listdir(folder_path)
image_files = [file for file in files if file.endswith((".jpg", ".jpeg", ".png"))]


for image_file in image_files:
    
    image_path = os.path.join(folder_path, image_file)
    image_name = os.path.basename(image_file)
    new_frame_time = time.time()
    img =cv2.imread(image_path)
    print(img.shape)
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Crop the detected face region from the original image
            face_roi = img[y1:y2, x1:x2]
            sharpness = compute_blurriness(face_roi)
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

                cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
                                   colorB=color)
                cvzone.putTextRect(img, f'Sharpness:  {int(sharpness)}',
                                   (max(0, x1), max(35, y2)), scale=2, thickness=4,colorR=color,
                                   colorB=color)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    output_folder='output'
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, img)
    cv2.waitKey(1)
