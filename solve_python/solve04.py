import cv2
import numpy as np
from solve02 import RGB2GRAY


# Q_4
def binarize_otsu(img):
    """Binarize image with otsu's method
    """
    img_gray = RGB2GRAY(img)
    pixels = img_gray.ravel()

    sigma_b_max = 0
    th = 0
    for i in range(256):
        black = pixels[pixels >= i]
        white = pixels[pixels < i]
        n_b = len(black)
        n_w = len(white)
        m_b = np.mean(black)
        m_w = np.mean(white)

        # variance between class
        sigma_b = n_b * n_w / ((n_b + n_w) ** 2) * ((m_b - m_w) ** 2)
        if sigma_b > sigma_b_max:
            th = i
            sigma_b_max = sigma_b

    img_b = np.where(img_gray < th, 0, 255)
    return img_b


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = binarize_otsu(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_04.png", img_out)


if __name__ == '__main__':
    main()
