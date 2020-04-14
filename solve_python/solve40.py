import cv2
import numpy as np
from solve36 import dct, idct
from solve38 import quantile
from solve39 import RGB2YCbCr, YCbCr2RGB


def main():
    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
               (12, 12, 14, 19, 26, 58, 60, 55),
               (14, 13, 16, 24, 40, 57, 69, 56),
               (14, 17, 22, 29, 51, 87, 80, 62),
               (18, 22, 37, 56, 68, 109, 103, 77),
               (24, 35, 55, 64, 81, 104, 113, 92),
               (49, 64, 78, 87, 103, 121, 120, 101),
               (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    img_yvbcr = RGB2YCbCr(img_in)
    F = dct(img_yvbcr)

    F = quantile(F, Q1, 0)
    F = quantile(F, Q1, 1)
    F = quantile(F, Q2, 2)
    img_ = YCbCr2RGB(idct(F, K=8))
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_40.jpg", img_out)

if __name__ == '__main__':
    main()
