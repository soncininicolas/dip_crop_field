# Code authored by: Javier Cremona and Nicolas Soncini
# You should have received a copy of the The MIT License along with this
# program

"""
Code to perform a simple k-means sky segmentation
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Any, Tuple
import bounding_box as bbox 


# Connected intensity components with stats
# Same as connectedComponentsWithStats but intensity-based
# that is: all contiguous pixels with the same intensity
# get clustered into a unique component
def connectedGrayscaleComponentsWithStats(image: cv2.Mat) -> \
        Tuple[int, cv2.Mat, np.ndarray, np.ndarray]:
    """
    Returns grayscale based components in an image and some properties.
    It imitates cv2's connectedComponentsWithStats but it clusters all
    contiguous pixels with the same intensity into a single component.

    Arguments:
        image (cv2.Mat): the input image in grayscale values [0,255]

    Returns:
        (Tuple[
            int        : the number of labels found 
            cv2.Mat    : an image with each pixeled labeled
            np.ndarray : the stats of each component (see original fn)
            np.ndarray : the centroid of each component (see original fn)
        ) 
    """
    retval = 0
    labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    stats = np.array([[0,0,0,0,0]])
    centroids = np.array([[0,0]])
    for i in range(image.max()+1):
        # binarize image
        bimg = (image == i).astype(np.uint8)
        # compute components with stats
        ival, ilab, ista, icen = cv2.connectedComponentsWithStats(bimg)
        if ival < 2:
            continue
        ilab[ilab.nonzero()] += retval # we shift the labels values
        retval = retval + (ival - 1)  # ignore background label
        labels = labels + ilab.astype(np.uint32)
        stats = np.concatenate((stats, ista[1:,:]))
        centroids = np.concatenate((centroids, icen[1:,:]))
    return retval, labels, stats, centroids


# Cluster Detection
# join clusters to minimize the number of pixels that allows it to span
# all the way from left to right
def horizon_cluster(clusters: cv2.Mat) -> Tuple[List[int], cv2.Mat]:
    """
    Takes in an image, segments it into clusters and separates all 
    disconnected clusters to returns a binary image representing the
    above-horizon portion of the image.
    
    Arguments:
        clusters (cv2.Mat): input image

    Returns:
        Tuple[
            List[int]: list of labels used for the above-horizon
            cv2.Mat  : mask of the above-horizon part of the image
        ]
    """
    crval, clabels, cstats, _ = connectedGrayscaleComponentsWithStats(clusters)
    # NOTE: We change things here, we organize the clusters from top most
    # to bottom most and we start joining them in this order until we get
    # a super-cluster that spans all the width of the image

    # Sort components by their top (y-coordinate) position
    elements = [(lab, 
                 cstats[lab, cv2.CC_STAT_LEFT], 
                 cstats[lab, cv2.CC_STAT_TOP], 
                 cstats[lab, cv2.CC_STAT_WIDTH], 
                 cstats[lab, cv2.CC_STAT_HEIGHT]) 
                for lab in range(1, crval+1)]  # Ignore the background label (0)
    elements.sort(key=lambda e: e[2])  # Sort by top (y-coordinate)

    # Join labels until they span the full width
    jlabels = np.zeros((clabels.shape[0], clabels.shape[1]), dtype=bool)
    rvals = []
    sky_elements = []
    for elem in elements:
        l, x, y, w, h = elem
        rvals.append(l)
        sky_elements.append(elem)
        jlabels[clabels == l] = True
        # check if it spans the full image width
        if jlabels.cumsum(axis=0, dtype=bool)[-1].all():
            break
    
    # Add additional labels if they contribute to the sky
    # calculate the sky's compound bounding box
    sky_bbox = bbox.merge_bounding_boxes(sky_elements)
    other_rvals = []
    for ol, ox, oy, ow, oh in elements:
        if ol in rvals:
            continue  # skip

        other_bbox = (ox, oy, ow, oh)

        # if bbox.intersects_with_tolerance(sky_bbox, other_bbox):
        #     other_rvals.append(other_lab)
        #     jlabels[clabels == other_lab] = True

        if bbox.is_contained(sky_bbox, other_bbox, 0.1):
            other_rvals.append(ol)
            jlabels[clabels == ol] = True

        # # check if vertically close to any sky component
        # for sky_lab in rvals:
        #     if (cstats[other_lab, cv2.CC_STAT_TOP] + 
        #             cstats[other_lab, cv2.CC_STAT_HEIGHT] <= 
        #             cstats[sky_lab, cv2.CC_STAT_TOP] + 
        #             cstats[sky_lab, cv2.CC_STAT_HEIGHT]):
        #         other_rvals.append(other_lab)
        #         jlabels[clabels == other_lab] = True
        #         break
    rvals = rvals + other_rvals  # so we dont mess up the rvals iteration

    return rvals, jlabels


def KMeansSky(image: cv2.Mat, **kwargs) -> cv2.Mat:
    """
    Takes an image in BGR format and returns a mask that represents
    the pixels that belong to the sky. Assumes that there's sky
    present in the image, that its positioned at the top of the image
    and that it spans the full image width.
    """
    ksize = kwargs['ksize'] if 'ksize' in kwargs else (25,25)
    sigmaX = kwargs['sigmaX'] if 'sigmaX' in kwargs else 0
    imblur = cv2.GaussianBlur(image, ksize=ksize, sigmaX=sigmaX)
    imkmean = imblur.reshape((-1, 3))
    imkmean = np.float32(imkmean)
    # perform kmeans clustering
    k = kwargs['k'] if 'k' in kwargs else 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(imkmean, k, None, criteria, 10, flags)
    # retransform to image
    centers = np.uint(centers)
    imkclus = centers[labels.flatten()]
    imkclus = imkclus.reshape((imblur.shape))
    imkclusg = cv2.cvtColor(imkclus.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hclabs, hcintens = horizon_cluster(imkclusg)
    return imkclusg, hcintens


def IntensitySky(image: cv2.Mat, **kwargs) -> cv2.Mat:
    """
    Takes an image in BGR format and returns a mask that represents
    the pixels that belong to the sky. Assumes that there's sky
    present in the image, that its positioned at the top of the image
    and that it spans the full image width.
    """
    ksize = kwargs['ksize'] if 'ksize' in kwargs else (25,25)
    sigmaX = kwargs['sigmaX'] if 'sigmaX' in kwargs else 0
    imblur = cv2.GaussianBlur(image, ksize=ksize, sigmaX=sigmaX)
    imintens = cv2.cvtColor(imblur, cv2.COLOR_BGR2GRAY)
    # perform quantization
    k = kwargs['k'] if 'k' in kwargs else 5
    q = 255 // k
    imquant = (imintens // q)
    hclabs, hcintens = horizon_cluster(imquant)
    return imquant, hcintens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', type=Path, required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)
    
    # process image for sky detection
    intensity_args = {
        'ksize': (15,15),
        'sigmaX': 0,
        'k': 3
    }
    intensity_clusters, intensity_sky = IntensitySky(image, **intensity_args)
    _, intensity_labels, _, _ = connectedGrayscaleComponentsWithStats(intensity_clusters)

    kmeans_args = {
        'ksize': (15,15),
        'sigmaX': 0,
        'k': 3
    }
    kmeans_clusters, kmeans_sky = KMeansSky(image, **kmeans_args)
    _, kmeans_labels, _, _ = connectedGrayscaleComponentsWithStats(kmeans_clusters)


    # convert to format opencv can display
    intensity_sky = (intensity_sky * 255).astype(np.uint8) 
    kmeans_sky = (kmeans_sky * 255).astype(np.uint8)

    # display
    cv2.imshow("Intensity Labels", intensity_labels.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow("Intensity-based Sky Detection", intensity_sky)
    cv2.waitKey(0)
    cv2.imshow("KMeans Labels", kmeans_labels.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow("KMeans-based Sky Detection", kmeans_sky)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
