import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TO DO:
#   * add more functionality

# Read the file in color mode.
img = cv.imread('images/tomato2.jpg', cv.IMREAD_COLOR)

# The default color space is BGR, we have to switch to RGB.
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)      # RGB to display
hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)  # HSV to segment

# Segment blue regions
lower_bound = (90, 100, 100)    # Hue is between 90 and 110
upper_bound = (110, 255, 255)
mask = cv.inRange(hsv_img, lower_bound, upper_bound)
result = cv.bitwise_and(img, img, mask=mask)

# Display result with Matplotlib
plt.figure(num=None, figsize=(14, 6), dpi=80, facecolor='gray', edgecolor='k')
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
