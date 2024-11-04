import cv2
import numpy as np

def merge_bounding_boxes(stats, selected_labels):
    """
    Merges bounding boxes for selected components based on their labels.

    Arguments:
        stats (np.ndarray): the output from cv2.connectedComponentsWithStats.
        selected_labels (List[int]): list of labels for components to merge.

    Returns:
        merged_box (
            Tuple[
                int: x,
                int: y,
                int: width,
                int: height]
            ): merged bounding box.
    """
    # Initialize min and max coordinates for merging
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    
    # Iterate over selected labels to find the min and max x, y coordinates
    for lab in selected_labels:
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]
        
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    
    # Calculate merged bounding box dimensions
    merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)
    return merged_box


def intersects_with_tolerance(box1, box2, tol=0.9):
    """
    Checks if two bounding boxes intersect within a given tolerance.

    Parameters:
        box1, box2 (Tuple[...]):  bounding boxes in (x, y, width, height)
        tol (float): tolerance percent (0 < tol < 1)

    Returns:
        (bool): True if the boxes intersect within the tolerance
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the area of both boxes
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Find the intersection rectangle coordinates
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Calculate width and height of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # Calculate intersection area
    inter_area = inter_width * inter_height

    # Check if the intersection area meets the percentage requirement
    if inter_area >= tol * area_box1 or inter_area >= tol * area_box2:
        return True
    return False


def is_contained(box1, box2, tol=0.1):
    """
    Checks if box2 is entirely contained within box1, with tolerance
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the absolute tolerance based on box1's dimensions
    x_tolerance = tol * w1
    y_tolerance = tol * h1

    # Calculate the expanded boundaries of box1 with relative tolerance
    box1_left, box1_right = x1 - x_tolerance, x1 + w1 + x_tolerance
    box1_top, box1_bottom = y1 - y_tolerance, y1 + h1 + y_tolerance

    # Calculate the boundaries of box2
    box2_left, box2_right = x2, x2 + w2
    box2_top, box2_bottom = y2, y2 + h2

    # Check if all edges of box2 are within the expanded boundaries of box1
    if (box2_left >= box1_left and box2_right <= box1_right and
        box2_top >= box1_top and box2_bottom <= box1_bottom):
        return True
    return False