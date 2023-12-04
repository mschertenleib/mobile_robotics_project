import threading
import time

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


def main():
    frame_width = 960
    frame_height = 720
    video_thread = VideoThread(frame_width, frame_height)

    frame = np.empty((frame_height, frame_width, 3), dtype=np.uint8)

    last_frame_time = time.time()
    while True:
        frame_start = time.time()
        delta_t = frame_start - last_frame_time
        last_frame_time = frame_start
        if delta_t > 0:
            print(f'{delta_t * 1000:10.4f} ms, {1 / delta_t:8.2f} fps')

        video_thread.get_frame(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    video_thread.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
