import cv2
import numpy as np
from solve36 import dct, idct


def quantile(F, Q, c, T=8):
    H, W, _ = F.shape
    for ys in range(0, H, T):
        for xs in range(0, W, T):
            F[ys: ys + T, xs: xs + T, c] = np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q

    return F


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()

    K = 8
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
              (12, 12, 14, 19, 26, 58, 60, 55),
              (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62),
              (18, 22, 37, 56, 68, 109, 103, 77),
              (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    F = dct(img_in)
    for i in range(3):
        F = quantile(F, Q, i)
    img_ = idct(F, K=K)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_38.jpg", img_out)

if __name__ == '__main__':
    main()
