import cv2 as cv
import numpy as np
import time
from playsound import playsound

def calibration():
    import glob

    cap = cv.VideoCapture(1)
    db=0
    inp = input("chessboardEdgeLength (default=0.0254): ")
    if inp == "" or inp in "dD" or inp.lower() == "default":
        chbEdgeLength = 0.0254
    else:
        chbEdgeLength = float(inp)

    font = cv.FONT_HERSHEY_SIMPLEX
    start=True
    calib=False

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*chbEdgeLength
    
    # arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    while True:
        _,frame=cap.read()

        if calib==False:
            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            # find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (9,6),None)
        
        if db<20:
            # if found, add object points, image points (after refining them)
            if ret == True and start:
                playsound('images/calibration/beep.mp3')
                start=False
                tstart=time.time()
                objpoints.append(objp)
            
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
            
                # draw and display the corners
                frame = cv.drawChessboardCorners(frame, (9,6), corners2,ret)
                cv.imshow('frame',frame)
                db=db+1
            elif ret == True and time.time()-tstart>0.5:
                tstart=time.time()
                objpoints.append(objp)
            
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
            
                # draw and display the corners
                frame = cv.drawChessboardCorners(frame, (9,6), corners2,ret)
                cv.imshow('frame',frame)
                db=db+1
            else:
                if ret==True:
                    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    frame = cv.drawChessboardCorners(frame, (9,6), corners2,ret)
                else:                
                    cv.putText(frame, "Please show chessboard.", (0,64), font, 1, (0,0,255),2,cv.LINE_AA)
                
                cv.imshow('frame',frame)
        else:
            if calib==False: # save the camera matrices first
                playsound('images/calibration/beep.mp3')
                calib=True
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                h, w = frame.shape[:2]
                newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                np.savez("camcalib", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            
            # undistort
            udst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            # crop the image
            x,y,w,h = roi
            udst = udst[y:y+h, x:x+w]
            cv.putText(udst, "Camera calibrated.", (0,64), font, 1, (0,255,0),2,cv.LINE_AA)
            cv.imshow('frame',udst)
        
        k=cv.waitKey(1)
        if k==27:
            break

    cap.release()
    cv.destroyAllWindows()
    return 0

if __name__=="__main__":
        calibration()