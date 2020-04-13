import cv2
import numpy as np
from solve20 import hist_of_pixels


# Q_28
def affine_conversion(img, a=1, b=1, A=0, tx=0, ty=0):
    H, W, C = img.shape

    A = np.radians(A)
    H_ = np.round(H * a).astype(np.int)
    W_ = np.round(W * b).astype(np.int)

    x_ = np.repeat(np.arange(H_), W_).reshape(H_, -1)
    y_ = np.tile(np.arange(W_), (H_, 1))

    x = (b*np.cos(A) * x_ + a*np.sin(A) * y_) / (a * b) - tx
    y = (a*np.cos(A) * y_ - b*np.sin(A) * x_) / (a * b) - ty

    is_black_x = np.where((x < 0) | (x >= H), True, False)
    is_black_y = np.where((y < 0) | (y >= W), True, False)

    out = np.zeros((H_, W_, 3))

    x = np.clip(np.round(x), 0, H-1).astype(np.int)
    y = np.clip(np.round(y), 0, W-1).astype(np.int)
    
    out = img[x, y].copy()
    
    out[is_black_x] = 0
    out[is_black_y] = 0
     
    return out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = affine_conversion(img_in, tx=-30, ty=30)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_28.png", img_out)


if __name__ == '__main__':
    main()
