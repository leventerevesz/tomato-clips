import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_markers():
    "Creates 4X4 ArUco Markers"
    # Stolen code from the ArUco tutorial
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    fig = plt.figure()
    nx = 4
    ny = 3
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        img = aruco.drawMarker(aruco_dict,i, 600)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        ax.axis("off")
    plt.savefig("results/ArUco_Markers.pdf")
    plt.show()

make_markers()