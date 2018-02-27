#
#
#

import cv2
import numpy as np
import threading


class main:

    """
    """

    frames = {}
    current_id = 0
    trackers = {}

    def __init__(self):

        """
        """

        self.camera = cv2.VideoCapture(0)
        self.frame_lock = threading.Lock()

        # initialize trackers

        # initialize graphics window

    def update(self):

        """
        """

        # get new frame
        self.frames.update({self.current_id: self.get_frame()})
        self.current_id += 1

        # only store the past 100 frames; clean up memory
        if(self.current_id > 100):
            del self.frames[self.current_id - 100]

        # display graphics window

    def get_frame(self):

        """
        """

        status, frame = self.camera.read()
        frame = cv2.cv2Color(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)

        return(frame)
