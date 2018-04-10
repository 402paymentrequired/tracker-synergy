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


def ddistance(x_1, x_2, **kwargs):

    """
    Compute the discrete Lp distance between two vectors.

    Parameters
    ----------
    x_1, x_2 : float[]
        Input vectors
    lp= : float
        Norm to compute; passed on to dnorm.

    Returns
    -------
    float
        Computed norm
    """

    assert len(x_1) == len(x_2)

    difference = []
    for i in range(len(x_1)):
        difference.append(abs(x_1[i] - x_2[i]))

    return(dnorm(difference, **kwargs))


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


def gaussian_convolve(input, sigma):

    gaussian = [
        (
            1 / (sigma * math.pow(2 * math.pi, 0.5)) *
            np.exp(-0.5 * ((x / sigma) ** 2))
        )
        for x in range(-sigma, sigma)]

    return(np.convolve(input, gaussian, mode="full"))


def sift_scene(img_target, img_scene, **kwargs):
    """
    Find matches of an image in a scene with SIFT and FLANN.

    Parameters
    ----------
    img_target : np.array
        Input single channel reference image
    img_scene : np.array
        Scene to find the target in
    max_ratio : float
        Maximum ratio between best match and second match distances;
        matches with a higher ratio are discarded
    output= : bool
        Displays output window if set to True
    """

    # Initialize and run SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp_target, des_target = sift.detectAndCompute(img_target, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

    # Run FLANN
    # configuration
    FLANN_INDEX_KDTREE = 1
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # get best two matches, return as array of array of best matches
    matches = flann.knnMatch(des_target, des_scene, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # Run ratio test
    output = []
    for i, (best, second) in enumerate(matches):
        if best.distance < 0.9 * second.distance:
            matchesMask[i] = [1, 0]
            output.append([best, second])

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(
        img_target, kp_target, img_scene, kp_scene,
        matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    # build return dictionary
    match_vectors = {
        "target":
            [kp_target[match[0].queryIdx].pt for match in output],
        "scene":
            [kp_scene[match[0].trainIdx].pt for match in output],
        "weights":
            [match[0].distance * match[0].distance / match[1].distance
                for match in output],
        "length": len(output)
    }

    return(match_vectors)


def weighted_bin(bin_width, data, **kwargs):

    """
    Split the data into discrete bins.

    Parameters
    ----------
    bin_width : float
        Width of each bin
    data : float[]
        Data array; all data must be positive
    weights= : float[]
        Weight array
    epsilon= : float
        All except epsilon of the data will be contained;
        the rest will be truncated

    Returns
    -------
    array
        Array containing the size of each bin.
    """

    # set up sorting differently depending on if weights are included
    if("weights" in kwargs):
        # create zip
        assert(len(kwargs["weights"]) == len(data))
        sortarray = zip(data, kwargs["weights"])
        # find total weight
        total_weight = 0
        for weight in kwargs["weights"]:
            total_weight += weight
    else:
        sortarray = zip(data, [1] * len(data))
        total_weight = len(data)

    # set up epsilon, if included; defaults to 0
    if("epsilon" in kwargs):
        epsilon = kwargs["epsilon"]
    else:
        epsilon = 0

    # sort lists
    data, weights = zip(*sorted(sortarray))

    current_weight = 0
    current_bin = 0
    current_index = 0
    output_array = []
    # loop through until all but epsilon are included
    while(current_index < len(data) and current_weight < total_weight * (1.0 - epsilon)):

        # process data until it overflows to the next bin
        current_value = 0
        while(current_index < len(data) and data[current_index] < current_bin + bin_width):
            current_value += weights[i]
            current_weight += weights[i]
            current_index += 1

        # append to result
        output_array.append(current_value)
        current_bin += bin_width

    return(output_array)


if(__name__ == "__main__"):
    target = cv2.imread("reference/target2x0.5.jpg")
    scene = cv2.imread("reference/scenex0.125.jpg")
    weights = image_k_means(target, 1)

    match_vectors = sift_scene(
        weighted_grayscale(target, weights[0]),
        weighted_grayscale(scene, weights[0]))

    target = match_vectors["target"]
    scene = match_vectors["scene"]
    length = match_vectors["length"]
    weights = match_vectors["weights"]

    w = []
    hist = []
    for i in range(length):
        for j in range(i + 1, length):
            # filter out distances between points mapped to multiple others
            if(scene[i] != scene[j] and target[i] != target[j]):
                dist_ratio = (
                    ddistance(scene[i], scene[j]) /
                    (ddistance(target[i], target[j]) + 1))
                hist.append(dist_ratio)
                w.append(1 / (weights[i] * weights[j]))

    plt.hist(hist, bins=100, weights=w, range=[0, 5])
    plt.show()
    plt.plot(
        gaussian_convolve(
            weighted_bin(0.05, hist, weights=w, epsilon=0.2), 2))
    plt.show()

