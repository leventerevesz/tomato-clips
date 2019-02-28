import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

img = cv.imread('images/stem4.jpg',0)
edges = cv.Canny(img,10,400)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(edges,cv.CV_64F,1,0,ksize=9)
sobely = cv.Sobel(edges,cv.CV_64F,0,1,ksize=9)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(edges,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
#plt.subplot(324),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(326),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


plt.show()