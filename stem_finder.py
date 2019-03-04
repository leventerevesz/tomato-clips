import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

img = cv.imread('images/stem8.jpg')
img2=img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,0,400, apertureSize=3)
laplacian = cv.Laplacian(gray,cv.CV_64F)
sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=9)
sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=9)

# Hough lines
minLineLength=5
maxLineGap = 5
lines = cv.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
for line in lines:
    x1,y1,x2,y2=line[0]
    cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)

lines = cv.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(img)
plt.title('Probabilistic'), plt.xticks([]), plt.yticks([])
#plt.subplot(324),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2,cmap = 'gray')
plt.title('Normal'), plt.xticks([]), plt.yticks([])
#plt.subplot(326),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


plt.show()