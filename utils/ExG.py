# Code authored by: Javier Cremona and Nicolas Soncini
# You should have received a copy of the The MIT License along with this
# program

"""
Code to perform a Green Chromatic Excess Index (2g-r-b index, add cite)
"""
import cv2
import numpy as np

def ExG(image: cv2.Mat) -> cv2.Mat:
    """
    Takes image in BGR format and returns the green chromatic excess
    index applied to the image in the form of a grayscale image
    in the [0, 1] range.
    """
    B, G, R = cv2.split(image)
    N = (R + B + G).astype('float') + np.ones_like(R).astype('float')
    r = np.divide(R.astype('float'), N)
    g = np.divide(G.astype('float'), N)
    b = np.divide(B.astype('float'), N)
    exg = (2 * g) - r - b
    exg =  (exg * 255).astype('uint8')
    return exg