import cv2
import numpy as np


# Q_36
def C(u):
    if u == 0:
        return 1 / np.sqrt(2)
    else:
        return 1


def dct(img, T=8):
    H, W, channels = img.shape

    F = np.zeros_like(img, dtype=np.float64)
    for x_ in range(0, W, T):
        for y_ in range(0, H, T):
            for u in range(T):
                for v in range(T):
                    for x in range(T):
                        for y in range(T):
                            for c in range(channels):
                                F[y_+v, x_+u, c] += 2 / T * C(u) * C(v) * img[y_+y, x_+x, c] * np.cos((2*x + 1) * u * np.pi / 2 / T) * np.cos((2*y + 1) * v * np.pi / 2 / T)
    return F


def idct(F, T=8, K=8):
    H, W, channels = F.shape
    
    f = np.zeros_like(F, dtype=np.float64)
    for c in range(channels):
        for x_ in range(0, W, T):
            for y_ in range(0, H, T):
                for x in range(T):
                    for y in range(T):
                        for u in range(K):
                            for v in range(K):
                                f[y_+y, x_+x, c] += 2 / T * C(u) * C(v) * F[y_+v, x_+u, c] * np.cos((2*x + 1) * u * np.pi / 2 / T) * np.cos((2*y + 1) * v * np.pi / 2 / T)
    return f


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    F = dct(img_in)
    print("dct done!")
    img_ = idct(F)
    print("idct done!")
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_36.jpg", img_out)


if __name__ == '__main__':
    main()
