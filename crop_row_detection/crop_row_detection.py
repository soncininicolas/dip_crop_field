import argparse
import cv2
import sys
import numpy as np
import math

def read_image(image_path):
    """Reads an image from the specified path.
    """  
    image = cv2.imread(image_path)
    # Check if image was read successfully
    if image is None:
        print(f"Error: Could not read image from path: {image_path}")
        sys.exit(1)
        
    print(f"Image successfully read from: {image_path}")
    return image

def ExGR(image: cv2.Mat) -> cv2.Mat:
    print("Index: ExGR")
    image = image.astype('float') / 255
    B, G, R = cv2.split(image)
    N = (R + B + G) + np.ones_like(R).astype('float') * 0.00001
    r = np.divide(R, N)
    g = np.divide(G, N)
    b = np.divide(B, N)
    exgr = np.maximum((2 * g) - r - b - (1.4 * r - g), np.zeros_like(R))
    exgr =  (exgr * 255).astype('uint8')
    return exgr

def ExG(image: cv2.Mat) -> cv2.Mat:
    """
    Takes image in BGR format and returns the green chromatic excess
    index applied to the image in the form of a grayscale image
    in the [0, 1] range.
    """
    print("Index: ExG")
    image = image.astype('float') / 255
    B, G, R = cv2.split(image)
    N = (R + B + G) + np.ones_like(R).astype('float') * 0.00001
    r = np.divide(R, N)
    g = np.divide(G, N)
    b = np.divide(B, N)
    exg = (2 * g) - r - b
    # exg = np.maximum(exg, np.zeros_like(G))    
    
    # It should map to [0,255] (uint8)
    exg = cv2.normalize(exg, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    exg =  (exg * 255).astype('uint8')
    return exg

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

if __name__ == "__main__":
    # Parse command line arguments
    # TODO complete description
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    
    image = read_image(args.image_path)
    exg = ExG(image)
    ret, otsu = cv2.threshold(exg, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    window_name = "ExG + Otsu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, otsu)
    cv2.waitKey(0)

    skel = skeleton(otsu, (5,5))
    window_name = "Skeleton"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, skel)
    cv2.waitKey(0)

    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(skel, cv2.MORPH_OPEN, kernel)
    # window_name = "Opening"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(window_name, opening)
    # cv2.waitKey(0)

    lines = cv2.HoughLines(skel, 1, np.pi / 180, 150, None, 0, 0)
    
    
    # Copy edges to the images that will display the results in BGR
    # cdst = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    cdst = np.copy(image)
    
    c = 2000
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + c*(-b)), int(y0 + c*(a)))
            pt2 = (int(x0 - c*(-b)), int(y0 - c*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    
    window_name = "Hough"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, cdst)
    cv2.waitKey(0)



    
    
