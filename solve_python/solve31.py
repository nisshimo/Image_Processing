import cv2
import numpy as np
from solve28 import affine_conversion


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    img_1 = affine_conversion(img_in, d=30)
    img_out_1 = np.clip(img_1, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_31_1.jpg", img_out_1)

    img_2 = affine_conversion(img_in, c=30)
    img_out_2 = np.clip(img_2, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_31_2.jpg", img_out_2)
    img_3 = affine_conversion(img_in, c=30, d=30)
    img_out_3 = np.clip(img_3, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_31_3.jpg", img_out_3)


if __name__ == '__main__':
    main()
