import time

from threaded_capture import *


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

        is_frame_new = video_thread.get_frame(frame)

        if delta_t > 0:
            print(f'{delta_t * 1000:10.4f} ms, {1 / delta_t:8.2f} fps, {is_frame_new = }')

        if is_frame_new:
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    video_thread.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
