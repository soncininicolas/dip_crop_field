import numpy as np
import cv2

def skeleton(image, kernel):
    skel = np.zeros((image.shape[0],image.shape[1]),np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,kernel)
    done = False
    size = np.size(image)
    while( not done):
        eroded = cv2.erode(image,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(image,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = eroded.copy()
    
        zeros = size - cv2.countNonZero(image)
        if zeros==size:
            done = True
    return skel

