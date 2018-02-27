#
#
#

import threading
import pygame


class tracker_base(threading.Thread):

    """
    """

    def __init__(self, center, size, frame_lock):

        """
        """
        self.name = "tracker name"

        threading.Thread.__init__(self)
        self.frame_lock = frame_lock

        self.current_frame = []
        self.central_frame = []

        self.center = center
        self.size = size
        self.confidence = 1.0

        self.clock = pygame.Clock.clock()

    def reinitialize(self, center, size, confidence):

        """
        """

        self.center = center
        self.size = size
        self.confidence = confidence

    def update(self):

        """
        """

        while(self.check_main_thread()):

            # wait to be serviced by central thread

            # update frame
            # set center, size, and confidence

            self.clock.tick()

    def check_main_thread(self):

        """
        Check if the main thread is alive. Should be used by all threads.

        Returns
        -------
        bool
            True if the main thread is alive; False otherwise
        """

        for thread in threading.enumerate():
            if(thread.name == "MainThread"):
                return(thread.is_alive())

        return(False)
