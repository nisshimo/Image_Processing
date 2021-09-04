import cv2
import numpy as np
from solve44 import hough_step1
from solve45 import hough_step2


# Q_46
def hough_step3(img, vote_table_maximal):
    
    H, W, C = img.shape
    R, T = vote_table_maximal.shape

    out = img.copy()
    for r in range(R):
        for t in range(T):
            if vote_table_maximal[r, t] > 0:
                t_r = np.deg2rad(t)
                for x in range(W):
                    y = np.int((-np.cos(t_r) * x + r - R / 2) / (np.sin(t_r) + 1e-10))
                    if 0 <= y < H:
                        out[y, x, 0] = 0
                        out[y, x, 1] = 0
                        out[y, x, 2] = 255

    return out


def hough(img, N=30):
    vote_table = hough_step1(img)
    vote_table_maximal = hough_step2(vote_table, N=N)
    out = hough_step3(img, vote_table_maximal)
    return out


def main():
    img_in = cv2.imread("../img/in/thorino.jpg").astype(np.float64).copy()
    out = hough(img_in, N=30)

    out_clip = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_46.jpg", out_clip)


if __name__ == '__main__':
    main()
