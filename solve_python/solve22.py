import cv2
import numpy as np
from solve20 import hist_of_pixels
import matplotlib.pyplot as plt


# Q_22
def hist_transform(img, m0=128, s0=52):
    pixels = img.ravel()
    m = np.mean(pixels)
    s = np.std(pixels)
    out = s0 / s * (img - m) + m0
    return out


def main():
    img_in = cv2.imread("../img/in/imori_dark.jpg").astype(np.float64)
    img_ = hist_transform(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_22_1.jpg", img_out)
    fig, ax = plt.subplots()
    hist_of_pixels(img_out, ax)
    plt.savefig("../img/out/q_22_2.jpg")


if __name__ == '__main__':
    main()
