# feature_tracker.py
# feature identification based tracker

import cv2
import numpy as np
from matplotlib import pyplot as plt


def weighted_grayscale(image, weight):

    """
    Get a weighted grayscale image.

    Parameters
    ----------
    image : np.array
        Input BGR image
    weight : array
        [B,G,R] weights

    Returns
    -------
    np.array
        Weighted grayscale image
    """

    weight = [x * 255 / max(weight) for x in weight]
    split_image = cv2.split(image)
    return(np.uint8(
        split_image[0] * (weight[0] / 255.) / 3 +
        split_image[1] * (weight[1] / 255.) / 3 +
        split_image[2] * (weight[2] / 255.) / 3
    ))


def image_k_means(image, k):

    """
    Get dominant colors from a target image.

    Parameters
    ----------
    image : np.array
        Input BGR image
    k : int
        Number of colors to extract

    Returns
    -------
    array
        List of [B,G,R] tuples representing the K dominant colors.
    """

    array = np.copy(image).reshape((-1, 3))
    array = np.float32(array)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    return(center)


def test_func(img_target, img_scene):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_target, None)
    kp2, des2 = sift.detectAndCompute(img_scene, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(
        img_target, kp1, img_scene, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    for i in range(10):
        print(matches[i][0].trainIdx)
        print(matches[i][1].trainIdx)


target = cv2.imread("target.jpg")
scene = cv2.imread("scene.jpg")
weights = image_k_means(target, 1)

print(weights)
test_func(
    weighted_grayscale(target, weights[0]),
    weighted_grayscale(scene, weights[0]))
#test_func(cv2.imread("target.jpg", 0), cv2.imread("scene.jpg", 0))
