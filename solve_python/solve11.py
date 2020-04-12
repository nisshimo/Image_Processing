import cv2
import numpy as np


def visualize(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


# Q_11
def smoothing_filter(img, stride=3):
    stride_range = range(int(-stride/2), int(stride/2+1))
    out = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                subregion = np.zeros((stride, stride))
                for x in stride_range:
                    for y in stride_range:
                        if 0 <= i + x < img.shape[0] and 0 <= j + y < img.shape[1]:
                            subregion[x+1][y+1] = img[i+x, j+y, k]
                out[i, j, k] = np.mean(subregion)
    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = smoothing_filter(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_11.png", img_out)


if __name__ == '__main__':
    main()
