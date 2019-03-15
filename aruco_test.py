import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)
with np.load('camcalib.npz') as X:
        mtx = X['mtx']
        dist = X['dist']

i=0
while True:
    _, frame = cap.read()
    
    #img = cv.imread('images/calibration/calib_11.jpg')
    h, w = frame.shape[:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('undistorted',dst)
    
    k=cv.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        i+=1
        cv.imwrite('images/aruco/ar_test'+str(i)+'.jpg',frame)
        continue
    
cap.release()
cv.destroyAllWindows()