import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TO DO:
#   * add more functionality

# Read the file in color mode.
img = cv.imread('images/tomato1.jpg', cv.IMREAD_COLOR)
blur = cv.GaussianBlur(img,(7,7),0) # blur for filtering out little blue pixels

# Get image width and height
width, height, channels = img.shape

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
blur = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
hsv_img = cv.cvtColor(blur, cv.COLOR_RGB2HSV)  # HSV to segment

# Segment blue regions
lower_bound = (90, 100, 100)    # Hue is between 90 and 110
upper_bound = (110, 255, 255)
rgb_lower_bound = (0, 110, 130)    # RGB limits
rgb_upper_bound = (70, 160, 200)
mask_rgb = cv.inRange(blur, rgb_lower_bound, rgb_upper_bound)
mask_hsv = cv.inRange(hsv_img, lower_bound, upper_bound)
result_rgb = cv.bitwise_and(img, img, mask=mask_rgb)
result_hsv = cv.bitwise_and(hsv_img, hsv_img, mask=mask_hsv)

reversemask=255-mask_hsv # invert mask for Blob Detection
rmask=reversemask

# Dilate (white ^) and Erode (black ^)
dkernel = np.ones((3,3),np.uint8)
ekernel = np.ones((5,5),np.uint8)
dilate = cv.dilate(reversemask,dkernel,iterations = 1)
erosion = cv.erode(dilate,ekernel,iterations = 1)

# Contuors with bounding rectangle
_, contours, _=cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

rectlist=[]
mode=2 # 1: perfect contour, 2: straight rectangle, 3: rotated rectangle
for cnt in contours:
    if mode==1:
        cv.drawContours(erosion, [cnt], 0, (100,100,100), 5)
    if mode==2:
        x,y,w,h = cv.boundingRect(cnt)
        if w > width-20 or h > height-20:
            continue
        else:
            rectlist.append((x,y,w,h))
            #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    if mode==3:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img,[box],0,(255,0,0),5)

# Filtering out smaller rectangles
for i in range (3):
    avgw=0
    avgh=0
    maxw=0
    maxh=0
    for rect in rectlist: # average rectangle size for filtering
        avgw+=rect[2]
        avgh+=rect[3]
        if maxw<rect[2]: maxw=rect[2]
        if maxh<rect[3]: maxh=rect[3]
    avgw=avgw/len(rectlist)
    avgh=avgh/len(rectlist)

    for rect in rectlist: # filter out too small rectangles
        w, h = rect[2], rect[3]
        if w<maxw-avgw or h<maxh-avgh:
            rectlist.remove(rect)

final=[]
for i in range(len(rectlist)): #bounding rectangles for every pair of object
    for j in range(i + 1, len(rectlist)):
        x1,y1,w1,h1=rectlist[i]
        x2,y2,w2,h2=rectlist[j]
        if abs(y1-y2)<avgh*4 and abs(x1-x2)<avgw*2:
            if x1==x2 and y1==y2:
                continue
            else:
                if x1<x2: x=x1
                else: x=x2
                if y1<y2: y=y1
                else: y=y2
                if x1+w1<x2+w2: w=x2+w2-x
                else: w=x1+w1-x
                if y1+h1<y2+h2: w=y2+h2-y
                else: h=y1+h1-y
                
                if (x,y,w,h) not in final:
                    final.append((x,y,w,h))

for rect in final:
    x,y,w,h=rect
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)


# Detect blobs
keypoints = detector.detect(reversemask)
im_with_keypoints = cv.drawKeypoints(reversemask, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#im_with_keypoints2 = cv.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Write keypoints to file
i=0
with open('results/keypoints.txt', 'w') as fw:
    for point in keypoints:
        x=round(point.pt[0],2)
        y=round(point.pt[1],2)
        s=round(point.size,2)
        i += 1
        fw.write('%i: x: %s\t y: %s\t diam: %s\n' %(i, x, y, s) )

# Save image (bounding rectangle and blob)
cv.imwrite('results/rect.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.imwrite('results/blobs.jpg', cv.cvtColor(im_with_keypoints, cv.COLOR_RGB2BGR))

# Display result with Matplotlib
plt.figure(num=None, figsize=(14, 6), dpi=80, facecolor='white', edgecolor='k')

#plt.subplot(2, 2, 1)
#plt.imshow(mask_rgb, cmap="gray")
#plt.subplot(2, 2, 2)
#plt.imshow(result_rgb)
#plt.subplot(1, 2, 1)
#plt.imshow(rmask)
#plt.subplot(1, 2, 2)
#plt.imshow(erosion)
plt.imshow(img)
plt.show()
