import cv2
import numpy as np
from solve20 import hist_of_pixels


# Q_26
def bilinear_interpolate(img, ax=1.5 , ay=1.5):
    # 難しかったので解答を参照した
    H, W, C = img.shape

    aH = int(ay * H)
    aW = int(ax * W)
    
    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    y = y / ay
    x = x / ax

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)

    dx = x - ix
    dy = y - iy

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + \
            (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = bilinear_interpolate(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_26.jpg", img_out)


if __name__ == '__main__':
    main()
