import cv2
import numpy as np


# Q_6
def reduce_color(img):
    """Color reduction processing
    """
    return (img // 64) * 64 + 32


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = reduce_color(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_06.jpg", img_out)


if __name__ == '__main__':
    main()
