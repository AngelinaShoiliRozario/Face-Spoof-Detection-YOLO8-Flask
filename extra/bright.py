import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(1)  # For Webcam

# Set desired resolution
cap.set(3, 640)
cap.set(4, 480)

# Brightness adjustment factor (increase to brighten the image)
brightness_factor = 1.2

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Split the HSV image into its components
        h, s, v = cv2.split(hsv)

        # Adjust the value (brightness) channel
        v = cv2.multiply(v, brightness_factor)

        # Clip the values to the range [0, 255]
        v = np.clip(v, 0, 255)

        # Merge the HSV components back together
        hsv = cv2.merge([h, s, v])

        # Convert the HSV image back to the BGR color space
        brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Display the brightened frame
        cv2.imshow("Brightened Image", brightened_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
