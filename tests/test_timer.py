import time
from threading import Timer


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class X:
    def __init__(self):
        self.x = 42


def callback(x):
    print(f'{x.x = }')
    x.x += 1


def main():
    x = X()
    timer = RepeatTimer(0.5, callback, args=[x])
    timer.start()
    time.sleep(2)
    x.x = -17
    time.sleep(2)
    timer.cancel()


if __name__ == '__main__':
    main()
