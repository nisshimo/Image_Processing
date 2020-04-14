import cv2
import numpy as np
from solve36 import dct, idct

# Q_37
def MSE(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


def PSNR(img1, img2, vmax=255):
    psmr = 10 * np.log10(vmax ** 2 / MSE(img1, img2))
    return psmr


def bitrate(K):
    return K ** 2 / 8


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()

    K = 4
    F = dct(img_in)
    img_ = idct(F, K=K)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_37.jpg", img_out)
    print("PSNR:", PSNR(img_in, img_out))
    print("bitrate:", bitrate(K=K))


if __name__ == '__main__':
    main()
