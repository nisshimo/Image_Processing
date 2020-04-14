import cv2
import numpy as np
from solve20 import hist_of_pixels
from solve28 import affine_conversion


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    img_1 = affine_conversion(img_in, a=0.8, b=1.3)
    img_out_1 = np.clip(img_1, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_29_1.jpg", img_out_1)
    img_2 = affine_conversion(img_in, a=0.8, b=1.3, tx=-30, ty=30)
    img_out_2 = np.clip(img_2, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_29_2.jpg", img_out_2)


if __name__ == '__main__':
    main()
