import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TO DO:
#   * add more functionality

cap = cv.VideoCapture(0)

i=0
while True:
    # Read the camera in color mode.
    _, img = cap.read()
    #img = cv.imread('images/tomato1.jpg', cv.IMREAD_COLOR)
    blur = cv.GaussianBlur(img,(7,7),0) # blur for filtering out little blue pixels

    # Get image width and height
    width, height, channels = img.shape

    hsv_img = cv.cvtColor(blur, cv.COLOR_BGR2HSV)  # HSV to segment

    # Segment blue regions
    lower_bound = (40, 40, 30)    # Hue is between 50 and 110
    upper_bound = (105, 255, 255)
    rgb_lower_bound = (0, 0, 0)    # RGB limits
    rgb_upper_bound = (30, 255, 255)
    mask_rgb = cv.inRange(blur, rgb_lower_bound, rgb_upper_bound)
    mask_hsv = cv.inRange(hsv_img, lower_bound, upper_bound)
    result_rgb = cv.bitwise_and(img, img, mask=mask_rgb)
    result_hsv = cv.bitwise_and(hsv_img, hsv_img, mask=mask_hsv)

    reversemask=255-mask_hsv
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
    if len(rectlist) != 0:
        for i in range (8):
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
            cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

    cv.imshow('rectangles',img)

    k=cv.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        i+=1
        cv.imwrite('results/rect_res_'+str(i)+'.jpg',img)
        continue

cap.release()
cv.destroyAllWindows()
