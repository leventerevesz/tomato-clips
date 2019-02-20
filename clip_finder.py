import cv2
import numpy as np
import math

# TO DO:
#   * add more functionality

img = cv2.imread('tomato.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('tomato',img)

cv2.waitKey(0)
cv2.destroyAllWindows()