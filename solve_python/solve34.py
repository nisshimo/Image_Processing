import cv2
import numpy as np
from solve33 import bandpass_filter
from solve32 import dft, idft


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    G = dft(img_in)
    G_ = bandpass_filter(G, r_low=0.1)
    img_ = idft(G_)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_34.png", img_out)


if __name__ == '__main__':
    main()
