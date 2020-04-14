import cv2
import numpy as np


def RGB2YCbCr(img):
    out = np.zeros_like(img, dtype=np.float64)

    R = img[..., 2]
    G = img[..., 1]
    B = img[..., 0]

    # Y
    out[..., 2] = 0.299 * R + 0.5870 * G + 0.114 * B
    # Cb
    out[..., 1] = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    # Cr
    out[..., 0] = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    return out


def YCbCr2RGB(img):
    out = np.zeros_like(img, dtype=np.float64)

    Y = img[..., 2]
    Cb = img[..., 1]
    Cr = img[..., 0]

    # R
    out[..., 2] = Y + (Cr - 128) * 1.402
    # Gb
    out[..., 1] = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    # B
    out[..., 0] = Y + (Cb - 128) * 1.7718

    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    img_YCbCr = RGB2YCbCr(img_in)
    img_YCbCr[..., 2] *= 0.7
    img_ = YCbCr2RGB(img_YCbCr)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_39.jpg", img_out)


if __name__ == '__main__':
    main()
