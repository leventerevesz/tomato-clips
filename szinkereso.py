# legszuperebb színkiválasztó.py

import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
#img = np.zeros((300,512,3), np.uint8)
image = cv2.imread(r"images/tomato3.jpg")
img = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
img_ = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
# cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('H_low','image',0,255,nothing)
cv2.createTrackbar('S_low','image',0,255,nothing)
cv2.createTrackbar('V_low','image',0,255,nothing)
cv2.createTrackbar('H_high','image',0,255,nothing)
cv2.createTrackbar('S_high','image',0,255,nothing)
cv2.createTrackbar('V_high','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    # itt már 
    if k == 27:
        break

    # get current positions of four trackbars
    H_low = cv2.getTrackbarPos('H_low','image')
    S_low = cv2.getTrackbarPos('S_low','image')
    V_low = cv2.getTrackbarPos('V_low','image')
    H_high = cv2.getTrackbarPos('H_high','image')
    S_high = cv2.getTrackbarPos('S_high','image')
    V_high = cv2.getTrackbarPos('V_high','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([H_low, S_low, V_low])
        upper_blue = np.array([H_high, S_high, V_high])
        #print(lower_blue)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(img_,img_, mask= mask)
        #cv.imshow('hsv',hsv)
        img[:] = res[:]

cv2.destroyAllWindows()