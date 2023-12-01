from threading import Timer
import time
import cv2


g_is_running = True


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval) and g_is_running:
            self.function(*self.args, **self.kwargs)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def callback():
        global g_is_running

        if not cap.isOpened():
            g_is_running = False

        ret, frame = cap.read()
        if ret:
            cv2.imshow('Frame', frame)
        else:
            print('Cannot read frame')

        if cv2.waitKey(1) & 0xff == 27:
            g_is_running = False

    timer = RepeatTimer(1 / 30.0, callback)
    timer.start()

    global g_is_running
    while g_is_running:
        time.sleep(1.0 / 60.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
