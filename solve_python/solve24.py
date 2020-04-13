import cv2
import numpy as np
from solve20 import hist_of_pixels
import matplotlib.pyplot as plt


# Q_24
def gamma_correction(img, c=1, g=2.2):
    img_ = img / 255
    out = (img_ / c) ** (1 / g) * 255
    return out


def main():
    img_in = cv2.imread("../img/in/imori_gamma.jpg").astype(np.float64)
    img_ = gamma_correction(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_24.png", img_out)


if __name__ == '__main__':
    main()
