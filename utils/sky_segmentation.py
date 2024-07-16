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
    for i in range(256):  # FIXME: range can be greater than this?!
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
    elements = [(lab, cstats[lab, cv2.CC_STAT_TOP]) for lab in range(crval)]
    elements.sort(key=lambda e: e[1])  # ascending given that x starts as 0 at the top
    jlabels = np.zeros((clabels.shape[0], clabels.shape[1]), dtype=bool)  # initial joined clusters is empty
    rvals = []
    for lab,_ in elements:
        rvals.append(lab)
        jlabels = jlabels | (clabels == lab)
        # check if it spans the full image width
        if jlabels.cumsum(axis=0, dtype=bool)[-1].all():
            # FIXME: this joins unconnected elements and small elements too
            # keep joining all elements that are above current
            alsojoin = [l for l,_ in elements if l not in rvals \
             and cstats[l, cv2.CC_STAT_TOP] + cstats[l, cv2.CC_STAT_HEIGHT]\
                  < cstats[lab, cv2.CC_STAT_TOP] + cstats[lab, cv2.CC_STAT_HEIGHT]]
            for l in alsojoin:
                jlabels = jlabels | (clabels == l)
            break
    return rvals, jlabels


def KMeansSky(image: cv2.Mat) -> cv2.Mat:
    """
    Takes an image in BGR format and returns a mask that represents
    the pixels that belong to the sky. Assumes that there's sky
    present in the image, that its positioned at the top of the image
    and that it spans the full image width.
    """
    imblur = cv2.GaussianBlur(image, ksize=(25,25), sigmaX=0)
    imkmean = imblur.reshape((-1, 3))
    imkmean = np.float32(imkmean)
    # perform kmeans clustering
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(imkmean, k, None, criteria, 10, flags)
    # retransform to image
    centers = np.uint(centers)
    imkclus = centers[labels.flatten()]
    imkclus = imkclus.reshape((imblur.shape))
    imkclusg = cv2.cvtColor(imkclus.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hclabs, hcintens = horizon_cluster(imkclusg)
    return hcintens


def IntensitySky(image: cv2.Mat) -> cv2.Mat:
    """
    Takes an image in BGR format and returns a mask that represents
    the pixels that belong to the sky. Assumes that there's sky
    present in the image, that its positioned at the top of the image
    and that it spans the full image width.
    """
    imblur = cv2.GaussianBlur(image, ksize=(25,25), sigmaX=0)
    imintens = cv2.cvtColor(imblur, cv2.COLOR_BGR2GRAY)
    k = 5
    q = 255 // k
    imquant = (imintens // q)
    cgrval, cglab, cgstats, _ = connectedGrayscaleComponentsWithStats(imquant)
    hclabs, hcintens = horizon_cluster(imquant)
    return hcintens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', type=Path, required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)
    
    # process image for sky detection
    intensity_sky = IntensitySky(image)
    kmeans_sky = KMeansSky(image)

    # convert to format opencv can display
    intensity_sky = intensity_sky.astype(np.uint8) * 255
    kmeans_sky = kmeans_sky.astype(np.uint8) * 255

    # display
    cv2.imshow("Intensity-based Sky Detection", intensity_sky)
    cv2.waitKey(0)
    cv2.imshow("KMeans-based Sky Detection", kmeans_sky)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
