# Code authored by: Javier Cremona and Nicolas Soncini
# You should have received a copy of the The MIT License along with this
# program

"""
Code to perform Local Image Intensity Equalization
"""
import cv2
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from typing import Tuple


def sliding_window_centered_old(image: np.ndarray, window: Tuple[int, int]):
    """
    Generates sliding windows on an image centered on every pixel.

    Arguments:
        image (np.ndarray): input image
        window (Tuple[int, int]): window size

    Returns
        (Generator[Tuple[
            int       : x-pixel where the window is centered on the full image
            int       : y-pixel where the window is centered on the full image
            np.ndarray: region of interest of the image (windowed view) 
        ]]) 
    """
    if window[0] % 2 == 0 or window[1] % 2 == 0:
        raise ValueError("El tamaño de la ventana debe ser expresable como 2x+1.")
    
    # Get image dimensions
    h, w = image.shape[:2]
    half_m, half_n = window[0] // 2, window[1] // 2

    # Slide a window across the image
    for y in range(h):
        for x in range(w):
            # Get window coordinates adjusted for centering
            start_x = max(x - half_m, 0)
            end_x = min(x + half_m + 1, w)
            start_y = max(y - half_n, 0)
            end_y = min(y + half_n + 1, h)

            yield (x, y, image[start_y:end_y, start_x:end_x])


def sliding_window_centered(
        image: np.ndarray, window: Tuple[int, int],
        start: Tuple[int, int] = (0,0), end: Tuple[int, int] = (0,0)):
    """
    Generates sliding windows on an image centered on every pixel.

    Arguments:
        image (np.ndarray): input image
        window (Tuple[int, int]): window size
        start (Tuple[int, int]): start centered here
        end (Tuple[int, int]): end centered here

    Returns
        (Generator[Tuple[
            int       : x-pixel where the window is centered on the full image
            int       : y-pixel where the window is centered on the full image
            np.ndarray: region of interest of the image (windowed view) 
        ]]) 
    """
    if window[0] % 2 == 0 or window[1] % 2 == 0:
        raise ValueError("El tamaño de la ventana debe ser impar")
    
    # Get image dimensions
    h, w = image.shape[:2]
    half_m, half_n = window[0] // 2, window[1] // 2

    start_h = start[0] if start else 0
    end_h = end[0] if end else h
    start_w = start[1] if start else 0
    end_w = end[1] if end else w

    # Slide a window across the image
    for y in range(start_h, end_h):
        for x in range(start_w, end_w):
            # Get window coordinates adjusted for centering
            start_x = max(x - half_m, 0)
            end_x = min(x + half_m + 1, w)
            start_y = max(y - half_n, 0)
            end_y = min(y + half_n + 1, h)

            yield (x-start_w, y-start_h, image[start_y:end_y, start_x:end_x])


def local_equalization(image: np.ndarray, window: Tuple[int, int]) \
        -> np.ndarray:
    """
    Performs a local "windowed" equalization of an image with a predefined
    window size

    Arguments:
        image (np.ndarray): 8-bit single channel input image
        window (Tuple[int, int]): window size of height and width (H,W)

    Returns
        (np.ndarray): output equalized image
    """
    half_m, half_n = window[0] // 2, window[1] // 2
    image_h, image_w = image.shape[:2]
    image_bor = cv2.copyMakeBorder(
        image, half_m, half_m, half_n, half_n, cv2.BORDER_REPLICATE)
    image_leq = np.zeros_like(image)
    for (x, y, s_window) in sliding_window_centered(
                image_bor, window=window, 
                start=(half_m, half_n), 
                end=(image_h+half_m, image_w+half_n)):
        window_eq = cv2.equalizeHist(s_window)
        image_leq[y,x] = window_eq[half_n, half_m]
    return image_leq
