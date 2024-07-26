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

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

if __name__ == "__main__":
    # Parse command line arguments
    # TODO complete description
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    
    image = read_image(args.image_path)
    exg = ExG(image)
    window_name = "ExG"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, exg)
    cv2.waitKey(0)


    ret, otsu = cv2.threshold(exg, 75, 255, cv2.THRESH_BINARY)
    window_name = "ExG + Otsu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, otsu)
    cv2.waitKey(0)

    skel = skeleton(otsu, (9,9))
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
    
    # TODO this value should be defined in function of w and h
    c = 2000
    num_lines = 0
    slope_thresh = 0.8
    slope_max = 1.0
    step = 0.05
    candidates = {}
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # print(f"{a} {b} {a/b}")
            if abs(a) > slope_thresh:
                print(a)
                # candidates.append(lines[i])
                category = int(np.sign(a) * (abs(a) - slope_thresh) // step)
                print(category)
                if not (category in candidates):
                    candidates[category] = [lines[i]]
                else:
                    candidates[category].append(lines[i])

                num_lines += 1
                # pt1 = (int(x0 + c*(-b)), int(y0 + c*(a)))
                # pt2 = (int(x0 - c*(-b)), int(y0 - c*(a)))
                # cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    print(candidates)
    categories = list(candidates.keys())
    line_intersections = []
    for i in range(len(categories) - 1):
        for j in range(1,len(categories)-i):
            for k in candidates[categories[i+j]]:
                for l in candidates[categories[i]]:
                    line_intersections.append(intersection(l,k))
    
    # TODO apply k-means to line_intersections

    window_name = "Hough"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, cdst)
    cv2.waitKey(0)



    
    
