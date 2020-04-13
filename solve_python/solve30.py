import cv2
import numpy as np
from solve20 import hist_of_pixels
from solve28 import affine_conversion

# Q_30
def rotate(img, a=1, b=1, A=0):
    
    rad_ = np.radians(45 + A)
    tx = 64 - np.cos(rad_) * 64 * np.sqrt(2)
    ty = 64 - np.sin(rad_) * 64 * np.sqrt(2)

    H, W, C = img.shape

    A = np.radians(A)
    H_ = np.round(H * a).astype(np.int)
    W_ = np.round(W * b).astype(np.int)

    x_ = np.repeat(np.arange(H_), W_).reshape(H_, -1)
    y_ = np.tile(np.arange(W_), (H_, 1))

    x = (b*np.cos(A) * (x_ - tx) + a*np.sin(A) * (y_ - ty)) / (a * b)
    y = (a*np.cos(A) * (y_ - ty) - b*np.sin(A) * (x_ - tx)) / (a * b)

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
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    img_1 = affine_conversion(img_in, A=30)
    img_out_1 = np.clip(img_1, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_30_1.png", img_out_1)

    img_2 = rotate(img_in, A=30)
    img_out_2 = np.clip(img_2, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_30_2.png", img_out_2)


if __name__ == '__main__':
    main()
