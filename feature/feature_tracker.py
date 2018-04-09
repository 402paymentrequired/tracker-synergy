# feature_tracker.py
# feature identification based tracker

import cv2
import math
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


def dnorm(vector, **kwargs):

    """
    Compute the discrete Lp norm of a vector.

    Parameters
    ----------
    vector : float[]
        Input vector
    lp= : float
        Norm to compute. Defaults to 2

    Returns
    -------
    float
        Computed norm
    """

    if("lp" in kwargs):
        lp = kwargs["lp"]
    else:
        lp = 2

    norm = 0
    for coord in vector:
        norm += math.pow(abs(coord), lp)

    return(math.pow(norm, 1.0 / lp))


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

    output = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
            output.append([m, n])

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(
        img_target, kp1, img_scene, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    match_vectors = {
        "target": [kp1[match[0].queryIdx].pt for match in output],
        "scene": [kp2[match[0].trainIdx].pt for match in output],
        "weights": [match[0].distance for match in output],
        "length": len(output)
    }

    return(match_vectors)


target = cv2.imread("target2.jpg")
scene = cv2.imread("scene2.jpg")
weights = image_k_means(target, 1)

print(weights)
match_vectors = test_func(
    weighted_grayscale(target, weights[0]),
    weighted_grayscale(scene, weights[0]))
# test_func(cv2.imread("target.jpg", 0), cv2.imread("scene.jpg", 0))

target = match_vectors["target"]
scene = match_vectors["scene"]
length = match_vectors["length"]
weights = match_vectors["weights"]

x = []
y = []
w = []
hist = []
for i in range(length):
    for j in range(i, length):
        dx = dnorm([target[i][0] - target[j][0], target[i][1] - target[j][1]])
        dy = dnorm([scene[i][0] - scene[j][0], scene[i][1] - scene[j][1]])
        x.append(dx)
        y.append(dy)
        if(dy / (dx + 1) < 20):
            hist.append(dy / (dx + 1))
            w.append(1 / (weights[i] * weights[j]))

plt.scatter(x, y)
plt.show()

plt.hist(hist, bins=100, weights=w)
plt.show()
