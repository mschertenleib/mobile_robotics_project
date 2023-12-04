import threading

import cv2
import numpy as np


class VideoThread:
    def __init__(self, frame_width: int, frame_height: int):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._frame = np.empty((frame_height, frame_width, 3), dtype=np.uint8)
        self._should_stop = False
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._read_frames)
        self._t.daemon = True
        self._t.start()

    def _read_frames(self):
        while self._cap.isOpened() and not self._should_stop:
            # This might take a "long" time (namely more than 100ms, depending on the VideoCapture parameters)
            ret, tmp_frame = self._cap.read()
            if not ret:
                print('Failed to read camera frame')
                break
            # Only lock for the short moment it takes to copy the data.
            with self._lock:
                self._frame[:] = tmp_frame

    def get_frame(self, frame: np.ndarray):
        # Only lock for the short moment it takes to copy the data.
        with self._lock:
            frame[:] = self._frame

    def stop(self):
        # NOTE: this relies on Python's builtin types having atomic assignment (which should be true for bool)
        self._should_stop = True
