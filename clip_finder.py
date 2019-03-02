import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TO DO:
#   * add more functionality

# Read the file in color mode.
img = cv.imread('images/tomato4.jpg', cv.IMREAD_COLOR)
img = cv.GaussianBlur(img,(7,7),0)

# Blob detection parameters
params = cv.SimpleBlobDetector_Params()

params.minThreshold = 0
params.maxThreshold = 256

# Filter by Area
params.filterByArea = True
params.minArea = 30

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
     
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5
     
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.5

# Filter by Distance between Blobs
params.minDistBetweenBlobs = 90

# OpenCV version check for BlobDetection
ver = (cv.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv.SimpleBlobDetector(params)
else:
    detector = cv.SimpleBlobDetector_create(params)

# The default color space is BGR, we have to switch to RGB.
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)      # RGB to display
hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)  # HSV to segment

# Segment blue regions
lower_bound = (90, 100, 100)    # Hue is between 90 and 110
upper_bound = (110, 255, 255)
rgb_lower_bound = (0, 110, 130)    # RGB limits
rgb_upper_bound = (70, 160, 200)
mask_rgb = cv.inRange(img, rgb_lower_bound, rgb_upper_bound)
mask_hsv = cv.inRange(hsv_img, lower_bound, upper_bound)
result_rgb = cv.bitwise_and(img, img, mask=mask_rgb)
result_hsv = cv.bitwise_and(hsv_img, hsv_img, mask=mask_hsv)

reversemask=255-mask_hsv # invert mask for Blob Detection

# Detect blobs
keypoints = detector.detect(reversemask)
im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Write keypoints to file
i=0
with open('blobs/keypoints.txt', 'w') as fw:
    for point in keypoints:
        x=round(point.pt[0],2)
        y=round(point.pt[1],2)
        s=round(point.size,2)
        i += 1
        fw.write('%i: x: %s\t y: %s\t diam: %s\n' %(i, x, y, s) )

# Save image with detected Blobs
cv.imwrite('blobs/result.jpg', cv.cvtColor(im_with_keypoints, cv.COLOR_RGB2BGR))

# Display result with Matplotlib
plt.figure(num=None, figsize=(14, 6), dpi=80, facecolor='gray', edgecolor='k')

#plt.subplot(2, 2, 1)
#plt.imshow(mask_rgb, cmap="gray")
#plt.subplot(2, 2, 2)
#plt.imshow(result_rgb)
#plt.subplot(2, 2, 3)
#plt.imshow(mask_hsv, cmap="gray")
#plt.subplot(2, 2, 4)
plt.imshow(im_with_keypoints)
plt.show()

#print(len(keypoints))
