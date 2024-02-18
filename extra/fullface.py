import numpy as np
import cv2

cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# Load the XML files for face, eye, mouth, and nose detection into the program
face_cascade = cv2.CascadeClassifier('./opencv-files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv-files/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./opencv-files/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('./opencv-files/haarcascade_mcs_nose.xml')

# Read the image for further editing
# image = cv2.imread('./luna-photos/normal.jpg')
while True:

    success, image = cap.read()
    # Convert the RBG image to a grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Identify the face using Haar-based classifiers
    faces = face_cascade.detectMultiScale(gray_image, 1.4, 4)

    # Iteration through the faces array and draw a rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

    # Identify the eyes, nose, and mouth using Haar-based classifiers
    eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray_image, 1.5, 11)
    nose = nose_cascade.detectMultiScale(gray_image, 1.3, 5)

    print(eyes)
    print(mouth)
    print(nose)

    # Check if any eyes are detected
    if len(eyes) > 0:
        print("Eyes are visible")
    else:
        print("Eyes are not visible")

    # Check if any nose is detected
    if len(nose) > 0:
        print("Nose is visible")
    else:
        print("Nose is not visible")

    # Check if any mouth is detected
    if len(mouth) > 0:
        print("Mouth is visible")
    else:
        print("Mouth is not visible")


    # Iteration through the eyes, nose, and mouth array and draw a rectangle
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    for (mx, my, mw, mh) in mouth:
        cv2.rectangle(image, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)

    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(image, (nx, ny), (nx+nw, ny+nh), (0, 255, 255), 2)

    # Show the final image after detection
    cv2.imshow('Face, eyes, mouth, and nose detected image', image)
    cv2.waitKey(1)
    # cv2.waitKey(0)
    # Show a successful message to the user
    print("Face, eye, nose, and mouth detection is successful")
