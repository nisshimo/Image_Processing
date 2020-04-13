import cv2
import numpy as np
from solve20 import hist_of_pixels
import matplotlib.pyplot as plt


# Q_23
def hist_equalize(img, z_max=255):
    pixels = img.ravel()
    S = len(pixels)
    
    out = img.copy()
    sum_h = 0
    for i in range(256):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    return out


def main():
    img_in = cv2.imread("../img/in/imori_dark.jpg").astype(np.float64)
    img_ = hist_equalize(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_23_1.png", img_out)
    fig, ax = plt.subplots()
    hist_of_pixels(img_out, ax)
    plt.savefig("../img/out/q_23_2.png")


if __name__ == '__main__':
    main()
