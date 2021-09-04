import cv2
import numpy as np
from solve02 import RGB2GRAY
from solve09 import gaussian_filter
from solve15 import sobel_filter


# Q_41
def canny_step1(img):
    img_gray = np.expand_dims(RGB2GRAY(img), axis=-1)
    img_ga = gaussian_filter(img_gray, stride=5, sigma=-1.4)
    fy, fx = sobel_filter(img_ga)
    fx = np.clip(fx, 0, 255).astype(np.float32)
    fy = np.clip(fy, 0, 255).astype(np.float32)
    edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    fx = np.maximum(fx, 1e-5)
    angle = np.rad2deg(np.arctan(fy / fx))

    def q(x):
        if -22.5 < x <= 22.5:
            return 0
        elif 22.5 < x <= 67.5:
            return 45
        elif 67.5 < x <= 112.5:
            return 90
        elif 112.5 < x <= 157.5:
            return 135

    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    angle = np.vectorize(q)(angle)

    return edge, angle


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    edge, angle = canny_step1(img_in)
    edge_out = np.clip(edge, 0, 255).astype(np.uint8)
    angle_out = np.clip(angle, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_41_1.jpg", edge_out)
    cv2.imwrite("../img/out/q_41_2.jpg", angle_out)


if __name__ == '__main__':
    main()
