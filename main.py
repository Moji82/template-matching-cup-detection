# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 08:40:20 2024

@author: Mojtaba Ahmadieh Khanesar
"""

import cv2
import numpy as np


def matchmaking(image2,template):

    # Initialize a dictionary to store match scores
    # Initialize a dictionary to store match scores
    # Initialize a dictionary to store match scores
    match_scores = {}
    
    # Resize image1 from 0.2 to 1.2
    for scale in np.arange(1, 2, 0.2):
        resized_image1 = cv2.resize(image1, None, fx=scale, fy=scale)
    
        # Perform template matching
        result = cv2.matchTemplate(image2, resized_image1, cv2.TM_CCOEFF_NORMED)
        match_score = np.max(result)
        match_scores[scale] = match_score
    
    # Find the best match scale
    best_scale = max(match_scores, key=match_scores.get)
    
    if best_scale>0.7:
    
	    best_match_score = match_scores[best_scale]
	    
	    # Get the location of the best match
	    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	    top_left = max_loc
	    h, w = resized_image1.shape
	    bottom_right = (top_left[0] + w, top_left[1] + h)
	    
	    # Draw a rectangle around the best match
	    image2=cv2.rectangle(image2, top_left, bottom_right, 255, 2)
	    
    return image2




def read_webcam():
    # Initialize the webcam (use 0 for the default camera)
    cap = cv2.VideoCapture(0)
    temple_file='C:\\bg1.png' # point to cup template
    template = cv2.imread(temple_file, cv2.IMREAD_GRAYSCALE)


    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image2=matchmaking(gray_frame,template)
        cv2.imshow("Best Match", image2)
        cv2.waitKey(30)

        # Display the frame

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()





# Call the function to read from the webcam
read_webcam()





