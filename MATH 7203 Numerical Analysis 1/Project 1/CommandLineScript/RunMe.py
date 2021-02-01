'''
Description: Runs program to demonstrate the level detection algorithm
created by modifying the standard Canny Edge Detection Algorithm

Input: none

Output: Marked up Video of Level Draining From a Jug as algorithm draws a line
at the water level for each frame

Uses: Math 7203 Mini Project 1 Video ( 1 min).mp4, GallowayMath7203MiniProject01.py,
Python Version 3.7.3, Matplotlib Version 3.0.3, CV2 (OpenCV) Version 4.2.0,
Numpy Version 1.16.2

Author: Josh Galloway
Version: 1.0
Date: 27 Feb 2020
'''

import cv2 as cv
from GallowayMath7203MiniProject01 import detectLevel

if __name__ == "__main__":
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	# code modified from Help page 
	# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
	
	# Video File Name
	vidFile = 'Math 7203 Mini Project 1 Video ( 1 min).mp4'
	print()
	print()
	print('==================================================================')
	print('Running Video: ', vidFile)
	print('Press q to quit')
	print('==================================================================')
	
	#Filter Constants
	SIGMA = .5     # x-sigma, is copied to y-sigma if y-sigma is not specified
	K_SIZE = (5,5)  # Kernel Window Size
	
	# Window Constants
	X,Y = 160,0
	H,W = 220,7
	d = (X,Y,H,W)
	
	# Double Thresholding Constants
	HIGH = 50
	LOW = 10
	
	# Run video
	cap = cv.VideoCapture(vidFile)
	cv.namedWindow('frame',cv.WINDOW_NORMAL)
	if not cap.isOpened():
	    print('Failed to Open',str(vidFile))
	while cap.isOpened():
	    ret, frame = cap.read()
	    # if frame is read correctly ret is True
	    if not ret:
	        print("Last frame received (stream end). Exiting ...")
	        break
	    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Mark Level Line On Frame with Student's Program
	    gray = detectLevel(gray,d,(HIGH,LOW),(SIGMA,K_SIZE))
	    cv.imshow('frame', gray)
	    cv.resizeWindow('frame',800,600)
	    if cv.waitKey(1) == ord('q'):
	        break
	cap.release()
	cv.destroyAllWindows()