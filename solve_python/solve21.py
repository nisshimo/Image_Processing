import cv2
import numpy as np
from solve20 import hist_of_pixels
import matplotlib.pyplot as plt


# Q_21
def hist_normalize(img, a=0, b=255):
    pixels = img.ravel()
    c = pixels.min()
    d = pixels.max()
    out = (b - a) / (d - c) * (img - c) + a
    return out


def main():
    img_in = cv2.imread("../img/in/imori_dark.jpg").astype(np.float64)
    img_ = hist_normalize(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_21_1.png", img_out)
    fig, ax = plt.subplots()
    hist_of_pixels(img_out, ax)
    plt.savefig("../img/out/q_21_2.png")


if __name__ == '__main__':
    main()
