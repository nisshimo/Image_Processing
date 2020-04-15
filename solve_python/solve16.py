import cv2
import numpy as np
from solve02 import RGB2GRAY


# Q_16
def prewitt_filter(img, stride=3, mode='v'):
    
    if len(img.shape) == 3:
        img = RGB2GRAY(img).copy()

    stride_range = range(int(-stride/2), int(stride/2+1))

    Kv = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Kh = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    out = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            subregion = np.zeros((stride, stride))
            for x in stride_range:
                for y in stride_range:
                    if 0 <= i + x < img.shape[0] and 0 <= j + y < img.shape[1]:
                        subregion[x+1][y+1] = img[i+x, j+y]
            assert mode in ['v', 'h']
            if mode == 'v':
                out[i, j] = np.sum(Kv * subregion)
            else:
                out[i, j] = np.sum(Kh * subregion)

    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)

    for mode in ['v', 'h']:
        img_ = prewitt_filter(img_in, mode=mode)
        img_out = np.clip(img_, 0, 255).astype(np.uint8)
        cv2.imwrite("../img/out/q_16_{}.jpg".format(mode), img_out)


if __name__ == '__main__':
    main()
