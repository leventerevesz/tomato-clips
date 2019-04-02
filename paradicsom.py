import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import cv2.aruco as aruco
import argparse

from marker_detection import marker_detection
from clip_finder_camera import clip_finder_camera
from calibration import calibration

parser = argparse.ArgumentParser(description='paradicsom.py parancs')
parser.add_argument("parancs", 
                    help="[md] marker detektálás, [csd] csipesz detektálás, [k] kamera kalibráció")
args = parser.parse_args()

if args.parancs.lower() == "md":
    print("Marker detektálás")
    marker_detection()
elif args.parancs.lower() == "csd":
    print("Csipesz detektálás")
    clip_finder_camera()
elif args.parancs.lower() == "k":
    print("Kamera kalibráció")
    calibration()
else:
    print("[HIBA] Ismeretlen parancs")
