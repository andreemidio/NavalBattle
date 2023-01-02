import time

import cv2
import numpy as np


def translate(dx=0, dy=0):
    M = np.eye(3)
    M[0:2, 2] = (dx, dy)
    return M


def scale(s=1, sx=1, sy=1):
    M = np.diag([s * sx, s * sy, 1])
    return M


def rotate(alpha=0):
    M = np.eye(3)
    M[0:2, 0:3] = cv2.getRotationMatrix2D(center=(0, 0), angle=-alpha / np.pi * 180, scale=1.0)
    return M


if __name__ == '__main__':
    img = cv2.imread("images/photo_match_preenchida.jpg")
    (height, width) = img.shape[:2]
    duration = 50
    fps = 4
    t0 = time.time()
    for t in np.arange(0, duration, 1 / fps):
        M = translate(width / 2, height / 2) @ rotate(2 * np.pi * t / 10 * 5) @ translate(dx=width / 4) @ rotate(
            2 * np.pi * t) @ scale(1 - 0.1 * t) @ translate(dx=-width / 4, dy=-height / 4)
        output = cv2.warpAffine(src=img, M=M[0:2], dsize=(width * 2, height * 2), flags=cv2.INTER_CUBIC)
        cv2.namedWindow("output", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("output", output)
        now = time.time()
        left = (t0 + t) - now
        key = cv2.waitKey(max(1, int(left * 1000)))
        if key in (13, 27): break  # Esc, Enter,

    cv2.destroyAllWindows()
    exit()
