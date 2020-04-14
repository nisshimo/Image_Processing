import cv2
import numpy as np


# Q_2
def RGB2GRAY(img):
    """Grayscale an image
    """
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    img_gray = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel
    return img_gray


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = RGB2GRAY(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_02.jpg", img_out)


if __name__ == '__main__':
    main()
