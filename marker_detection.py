import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import pandas as pd

markerEdge=0.0273  # ArUco marker edge length in meters

cap = cv.VideoCapture(1)

# Get the calibrated camera matrices
with np.load('camcalib.npz') as X:
        mtx = X['mtx']
        dist = X['dist']

while True:
    _, frame = cap.read()

    #img = cv.imread('images/aruco/distorted/ar_test4.jpg')    
    h, w = frame.shape[:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # Undistort
    udst = cv.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop image
    x,y,w,h = roi
    udst = udst[y:y+h, x:x+w]
    #cv.imshow('undistorted',udst)

    gray = cv.cvtColor(udst, cv.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # Detecting markers: get corners and IDs
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Font for text display
    font = cv.FONT_HERSHEY_SIMPLEX

    if np.all(ids != None):
        ### IDs found
        # Pose estimation with marker edge length
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerEdge, mtx, dist)
        
        for i in range(0, ids.size):
            aruco.drawAxis(udst, mtx, dist, rvec[i], tvec[i], 0.1)  # Draw axis
            
            # Z component of tvec to string
            transl=tvec.reshape(len(tvec),3)
            strg = str(ids[i][0]) + ' z=' + str(round(transl[i][2],3))

            # Writing text to 4th corner
            cv.putText(udst, strg, (int(corners[i][0][2][0]),int(corners[i][0][2][1])), font, 0.5, (255,0,0),1,cv.LINE_AA)

        # Draw square around the markers
        aruco.drawDetectedMarkers(udst, corners)

        # Write to txt
        #data=pd.DataFrame(data=tvec.reshape(len(tvec),3),columns=["tx","ty","tz"],index=ids.flatten())
        #data.index.name="markers"
        #data.sort_index(inplace=True)
        #np.savetxt("data.txt",data)
    else:
        ### No IDs found
        cv.putText(udst, "No Ids", (0,64), font, 1, (0,0,255),2,cv.LINE_AA)

    
    cv.imshow('markers', udst)
    if cv.waitKey(1) == 27: 
        break  # Esc to quit

cap.release()
cv.destroyAllWindows()