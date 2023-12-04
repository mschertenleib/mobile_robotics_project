import threading

import cv2
import numpy as np


class VideoThread:
    """
    A threaded handler for cv2.VideoCapture, that continuously reads frames from the camera, and allows the caller to
    just get the latest frame. This allows the video capture framerate to be completely independent from the looping
    rate of the code that needs the frames. Obviously, this means that get_frame() might return the same frame multiple
    times, if called several times within a short interval.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self._is_frame_new = True
        self._should_stop = False
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._read_frames)
        self._t.daemon = True
        self._t.start()

    def _read_frames(self):
        """
        This is the function that runs in its own thread
        """
        while self._cap.isOpened() and not self._should_stop:
            # This might take a "long" time (namely more than 100ms, depending on the VideoCapture parameters)
            ret, tmp_frame = self._cap.read()
            if not ret:
                print('Failed to read camera frame')
                break
            # Only lock for the short moment it takes to copy the data.
            with self._lock:
                self._frame[:] = tmp_frame
                self._is_frame_new = True

    def get_frame(self, dst_frame: np.ndarray) -> bool:
        """
        Retrieves the latest frame. Returns True if the retrieved frame is different than the last call to get_frame()
        """
        # Only lock for the short moment it takes to copy the data.
        with self._lock:
            dst_frame[:] = self._frame
            is_frame_new = self._is_frame_new
            self._is_frame_new = False
            return is_frame_new

    def stop(self):
        # NOTE: this relies on Python's builtin types having atomic assignment (which should be true for bool)
        self._should_stop = True
