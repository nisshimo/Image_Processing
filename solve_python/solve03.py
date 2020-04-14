import cv2
import numpy as np
from solve02 import RGB2GRAY


# Q_3
def binarize(img):
    """Binarize image
    """
    img_gray = RGB2GRAY(img)
    img_b = np.where(img_gray < 128, 0, 255)
    return img_b


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = binarize(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_03.jpg", img_out)


if __name__ == '__main__':
    main()
