import cv2
import numpy as np


# Q_9
def gaussian_filter(img, stride=3, sigma=1.3):
    H, W, C = img.shape

    K = np.zeros((stride, stride))
    stride_range = range(int(-stride/2), int(stride/2+1))
    for x in stride_range:
        for y in stride_range:
            K[x+1][y+1] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    K /= np.sum(K)

    out = img.copy()

    for i in range(H):
        for j in range(W):
            for k in range(C):
                subregion = np.zeros_like(K, dtype=np.float64)
                for x in stride_range:
                    for y in stride_range:
                        if 0 <= i + x < img.shape[0] and 0 <= j + y < img.shape[1]:
                            subregion[x+1][y+1] = img[i+x, j+y, k]
                out[i, j, k] = np.sum(subregion * K)
    return out


def main():
    img_in = cv2.imread("../img/in/imori_noise.jpg").astype(np.float64)
    img_ = gaussian_filter(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_09.jpg", img_out)



if __name__ == '__main__':
    main()
