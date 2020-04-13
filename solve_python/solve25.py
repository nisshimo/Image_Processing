import cv2
import numpy as np
from solve20 import hist_of_pixels


# Q_25
def NN_interpolate(img, a=1.5):
    length = int(img.shape[0] * a)
    out = np.zeros((length, length, 3))

    for i in range(length):
        for j in range(length):
            out[i-1, j-1, :] = img[round(i / a), round(j / a), :]

    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = NN_interpolate(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_25.png", img_out)


if __name__ == '__main__':
    main()
