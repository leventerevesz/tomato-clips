import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

lower = np.array([185, 40, 40]) # ( deg, %, % )
upper = np.array([210, 100, 100]) # ( deg, %, % )

# convert to OpenCV HSV format ([0:179, 0:255, 0:255])
multiplier = np.array([0.5, 2.55, 2.55])
lower_bound = np.round(lower * multiplier)
upper_bound = np.round(upper * multiplier)

print("lower bound: ", lower_bound)
print("upper bound: ", upper_bound)

# convert to matplotlib HSV ([0:1, 0:1, 0:1])
divider = np.array([360, 100, 100])
upper_bound_plt = upper / divider
lower_bound_plt = lower / divider

# Create two square matrices filled with these two colors
u_square = np.full((10, 10, 3), upper_bound_plt)
l_square = np.full((10, 10, 3), lower_bound_plt)

# Display with Matplotlib
ax = plt.subplot(1, 2, 1)
ax.set_title('lower')
plt.imshow(hsv_to_rgb(l_square))
ax = plt.subplot(1, 2, 2)
ax.set_title('upper')
plt.imshow(hsv_to_rgb(u_square))
plt.show()
