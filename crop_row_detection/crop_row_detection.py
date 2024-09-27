import argparse
import cv2
import sys
import numpy as np
import math
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TwoLines:
    line1: np.ndarray
    line2: np.ndarray

    @property
    def intersection(self):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = self.line1[0]
        rho2, theta2 = self.line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]


def apply_dbscan(data, eps=5, min_samples=2):
  """Detects outliers in a list of 2D points using DBSCAN.

  Args:
    data: A list of (N, 1, 2) elements representing 2D points.
    eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples: The number of points required to form a dense region.

  Returns:
    A list of labels, where -1 indicates an outlier and other values indicate cluster membership.
  """

  # Flatten the data to (N, 2) format
  flattened_data = [point[0] for point in data]

  # Create a DBSCAN object
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)

  # Fit the DBSCAN model to the data
  labels = dbscan.fit_predict(flattened_data)

  return labels

def apply_k_means(data, num_clusters):
  """Applies k-means clustering to a list of 2D points.

  Args:
    data: A list of (N, 1, 2) elements representing 2D points.
    num_clusters: The desired number of clusters.

  Returns:
    A list of cluster labels for each data point.
  """

  # Flatten the data to (N, 2) format
  flattened_data = [point[0] for point in data]

  # Create a KMeans object
  kmeans = KMeans(n_clusters=num_clusters)

  # Fit the k-means model to the data
  kmeans.fit(flattened_data)

  # Get the cluster labels for each data point
  labels = kmeans.labels_

  return labels

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

def points_from_line(rho, theta):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + c*(-b)), int(y0 + c*(a)))
    pt2 = (int(x0 - c*(-b)), int(y0 - c*(a)))
    return pt1, pt2

def category_from_line(theta, slope_thresh, step):
    a = math.cos(theta)
    return int(np.sign(a) * (abs(a) - slope_thresh) // step)

def choose_cluster_with_lower_dispersion(data, labels):
  """Chooses the cluster with the lowest dispersion.

  Args:
    data: A list of 2D points.
    labels: A list of cluster labels.

  Returns:
    The index of the cluster with the lowest dispersion.
  """

  unique_labels = np.unique(labels)
  weighted_variances = []
  for label in unique_labels:
    cluster_data = np.array([data[i] for i in range(len(data)) if labels[i] == label])
    centroid = np.mean(cluster_data, axis=0)
    variance = np.mean(np.sum((cluster_data - centroid) ** 2, axis=1))
    cluster_size = len(cluster_data)
    weighted_variance = variance / cluster_size
    weighted_variances.append(weighted_variance)

  return np.argmin(weighted_variances)
#   for label in unique_labels:
#     cluster_data = np.array([data[i] for i in range(len(data)) if labels[i] == label])
#     centroid = np.mean(cluster_data, axis=0)
#     variance = np.mean(np.sum((cluster_data - centroid) ** 2, axis=1))
#     variances.append(variance)

#   return np.argmin(variances)


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


    ret, otsu = cv2.threshold(exg, 75, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    window_name = "ExG + Otsu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, otsu)
    cv2.waitKey(0)

    # skel = skeleton(otsu, (7,7))
    # window_name = "Skeleton"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(window_name, skel)
    # cv2.waitKey(0)

    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    window_name = "Closing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, closing)
    cv2.waitKey(0)

    skel = skeleton(closing, (9,9))
    window_name = "Skeleton"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, skel)
    cv2.waitKey(0)

    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(skel, kernel)#cv2.morphologyEx(skel, cv2.MORPH_OPEN, kernel)
    window_name = "Dilation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, dilate)
    cv2.waitKey(0)

    lines = cv2.HoughLines(dilate, 1, np.pi / 180, 195, None, 0, 0) #FIXME threshold == 350 for zavalla2022
    # print(lines.shape)
    # if lines is not None:
    #     dst = np.copy(image)
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(dst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    #     window_name = "All lines"
    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #     cv2.imshow(window_name, dst)
    #     cv2.waitKey(0)


    # Copy edges to the images that will display the results in BGR
    # cdst = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    cdst = np.copy(image)
    
    # TODO this value should be defined in function of w and h
    c = 2000
    num_lines = 0
    slope_thresh = 0.8
    slope_max = 1.00000001
    step = 0.10
    candidates = {}
    half_num_cats = int((slope_max - slope_thresh) // step)
    num_cats = half_num_cats * 2
    if lines is not None:
        for i in range(0, len(lines)):
            # Line representation -> rho, theta
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Filter lines using slope_thresh
            if abs(a) > slope_thresh:
                
                # Discretize the space according to the slope
                category = category_from_line(theta, slope_thresh, step) #int(np.sign(a) * (abs(a) - slope_thresh) // step)
                
                if not (category in candidates):
                    candidates[category] = [lines[i]]
                else:
                    candidates[category].append(lines[i])

                num_lines += 1          

                # Plot each category with a different color      
                color = tuple(cv2.applyColorMap(np.uint8([[int(255 * (category + half_num_cats) / num_cats)]]), cv2.COLORMAP_JET).flatten().tolist())
                pt1, pt2 = points_from_line(rho, theta)
                cv2.line(cdst, pt1, pt2, color, 3, cv2.LINE_AA)
        window_name = "Filtered lines (by slope)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, cdst)
        cv2.waitKey(0)

    # Intersection of all lines, except those from the same category
    categories = list(candidates.keys())
    pair_of_lines = []
    for i in range(len(categories) - 1):
        for j in range(1,len(categories)-i):
            for k in candidates[categories[i+j]]:
                for l in candidates[categories[i]]:
                    pair_of_lines.append(TwoLines(l,k))
    
    pair_of_lines = np.array(pair_of_lines)
    
    # Estimate the coordinates of the intersections
    line_intersections = [l.intersection for l in pair_of_lines]

    # Apply DBSCAN to find clusters of intersections and potential outliers
    labels = apply_dbscan(line_intersections)
    
    print(labels)
    non_negative_values, counts = np.unique(labels[labels >= 0], return_counts=True)
    most_frequent_index = np.argmax(counts)
    most_frequent_label = non_negative_values[most_frequent_index]
    
    # Filter out outliers 
    non_outlier_indices = np.where(labels != -1)[0]
    flattened_data = [point[0] for point in line_intersections]
    non_outlier_data = [flattened_data[i] for i in non_outlier_indices]
    non_outlier_labels = labels[non_outlier_indices]
    # cluster_lower_dispersion = choose_cluster_with_lower_dispersion(non_outlier_data, non_outlier_labels)
    # print(f"low disp {cluster_lower_dispersion}")

    # Filter lines
    # Is this the best method to choose the vanishing point?
    indices = np.where(labels == most_frequent_label) 
    img_2 = np.copy(image)
    # print(pair_of_lines[indices])
    
    for p in pair_of_lines[indices]:
        theta1 = p.line1[0][1]
        theta2 = p.line2[0][1]
        # print(category_from_line(theta1, slope_thresh, step))
        # print(category_from_line(theta2, slope_thresh, step))
        pt1, pt2 = points_from_line(p.line1[0][0], p.line1[0][1])
        cv2.line(img_2, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
        pt1, pt2 = points_from_line(p.line2[0][0], p.line2[0][1])
        cv2.line(img_2, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)


    window_name = "Lines after clustering intersections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_2)
    cv2.waitKey(0)

    # Plot lines and non-outlier intersections
    # plt.imshow(cdst)
    # plt.scatter(np.array(non_outlier_data)[:, 0], np.array(non_outlier_data)[:, 1], c=non_outlier_labels, cmap='viridis', s=20)
    # plt.title("Non-Outlier Clusters")
    # plt.xlabel("X-coordinate")
    # plt.ylabel("Y-coordinate")
    # plt.show()



    
    
