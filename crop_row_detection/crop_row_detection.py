import argparse
import cv2
import sys
import numpy as np

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

if __name__ == "__main__":
    # Parse command line arguments
    # TODO complete description
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    
    image = read_image(args.image_path)
    exg = ExG(image)
    ret, otsu = cv2.threshold(exg, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    window_name1 = "ExG + Otsu"
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name1, otsu)
    cv2.waitKey(0)



    
    
