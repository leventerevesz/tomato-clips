import cv2 as cv
import numpy as np
import cv2.aruco as aruco

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

cap = cv.VideoCapture(1)
with np.load('camcalib.npz') as X:
        mtx = X['mtx']
        dist = X['dist']

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
    #cv.imshow('undistorted',dst)

    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    font = cv.FONT_HERSHEY_SIMPLEX #font for displaying text (below)


    if np.all(ids != None):

        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
        
        for i in range(0, ids.size):
            aruco.drawAxis(dst, mtx, dist, rvec[i], tvec[i], 0.1)  # Draw Axis
        aruco.drawDetectedMarkers(dst, corners) #Draw A square around the markers


        ###### DRAW ID #####
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv.putText(dst, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv.LINE_AA)


    else:
        ##### DRAW "NO IDS" #####
        cv.putText(dst, "No Ids", (0,64), font, 1, (0,255,0),2,cv.LINE_AA)

    
    cv.imshow('markers', dst)
    if cv.waitKey(1) == 27: 
        break  # esc to quit
    
cap.release()
cv.destroyAllWindows()