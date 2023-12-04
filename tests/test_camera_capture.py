import time

import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = np.empty((frame_height, frame_width, 3), dtype=np.uint8)
    rgba = np.empty((frame_height, frame_width, 4), dtype=np.float32)

    last_sample_time = time.time()
    while cap.isOpened():
        now = time.time()
        delta_t = now - last_sample_time
        last_sample_time = now
        if delta_t > 0:
            print(f'{delta_t * 1000:8.2f} ms, {1 / delta_t:8.2f} fps')

        ret, _ = cap.read(frame)
        if not ret:
            break
        rgba[:, :, 2::-1] = frame / 255.0
        rgba[:, :, 3] = 1.0
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
