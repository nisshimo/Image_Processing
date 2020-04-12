import cv2
import numpy as np


def visualize(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


# Q_1
def RGB2BGR(img):
    """Swap RGB in order of BGR
    """
    img_ = img[:, :, (2, 1, 0)]
    return img_


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = RGB2BGR(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_01.png", img_out)


if __name__ == '__main__':
    main()
