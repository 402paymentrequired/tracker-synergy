# feature_tracker.py
# feature identification based tracker

import cv2
# import numpy as np
from matplotlib import pyplot as plt


def getHueChannel(filename):

    image = cv2.cvtColor(cv2.imread(filename, 1), cv2.COLOR_BGR2HSV)
    return(cv2.split(image)[0])


def testFunc():

    # img_target = cv2.split(cv2.imread('target.jpg', mode))[color]
    # img_scene = cv2.split(cv2.imread('scene.jpg', mode))[color]
    img_target = getHueChannel("target.jpg")
    img_scene = getHueChannel("scene.jpg")

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

testFunc()
