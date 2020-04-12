import cv2
import numpy as np
from solve02 import RGB2GRAY


# Q_19
def LoG_filter(img, stride=5, sigma=3):
    if len(img.shape) == 3:
        img = RGB2GRAY(img).copy()
    K = np.zeros((stride, stride))
    stride_range = range(int(-stride/2), int(stride/2+1))
    for x in stride_range:
        for y in stride_range:
            K[x+1][y+1] = (x ** 2 + y ** 2 - 2 * sigma ** 2) / (2 * np.pi * sigma ** 6) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    K /= np.sum(K)

    out = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            subregion = np.zeros_like(K, dtype=np.float64)
            for x in stride_range:
                for y in stride_range:
                    if 0 <= i + x < img.shape[0] and 0 <= j + y < img.shape[1]:
                        subregion[x+1][y+1] = img[i+x, j+y]
            out[i, j] = np.sum(subregion * K)
    return out


def main():
    img_in = cv2.imread("../img/in/imori_noise.jpg").astype(np.float64)
    img_ = LoG_filter(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_19.png", img_out)


if __name__ == '__main__':
    main()
