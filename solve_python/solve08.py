import cv2
import numpy as np


# Q_7
def pooling(img, stride=8, method=np.max):
    """Average pooling
    """
    out = img.copy()
    for i in range(img.shape[0] // stride):
        for j in range(img.shape[1] // stride):
            h = i*stride
            v = j*stride
            h_ = min((i+1)*stride, img.shape[0])
            v_ = min((j+1)*stride, img.shape[1])
            represent = method(method(img[h:h_, v:v_, :], axis=1), axis=0).reshape(1, 1, 3)
            out[h:h_, v:v_, :] = np.tile(represent, (h_ - h, v_ - v, 1))
    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = pooling(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_08.png", img_out)


if __name__ == '__main__':
    main()
