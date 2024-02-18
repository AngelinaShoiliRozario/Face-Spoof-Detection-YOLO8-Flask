# import cv2
# import numpy as np
# import os

# # Read the image
# image = cv2.imread('./datasets/sharp9.jpg')


# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Calculate the histogram of pixel intensities
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# # Apply a threshold to the histogram
# threshold = 50
# reflection_mask = hist > threshold

# # Find contours of reflection areas
# contours, _ = cv2.findContours(reflection_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw contours on the original image
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Display the original image with reflections highlighted
# cv2.imshow('Reflection Detection', image)
# output_folder='reflected'
# output_path = os.path.join(output_folder, 'reflected.jpg')
# cv2.imwrite(output_path, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os

# Read the image
image = cv2.imread('./datasets/sharp9.jpg')


# Define the gamma value (adjust as needed)
gamma = 3

# Apply gamma correction to reduce brightness
corrected_image = np.uint8(cv2.pow(image / 255.0, gamma) * 255.0)

# Display the original and corrected images
# cv2.imshow('Original Image', image)
# cv2.imshow('Corrected Image', corrected_image)

output_folder='reflected'
output_path = os.path.join(output_folder, 'reflected.jpg')
cv2.imwrite(output_path, corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
