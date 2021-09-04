import cv2
import numpy as np
from solve44 import hough_step1


# Q_45
def hough_step2(vote_table, N=50):
    H, W = vote_table.shape

    out = np.zeros_like(vote_table, dtype=np.int)

    vote_array = vote_table.ravel().copy()
    sorted_vote_array = np.argsort(-vote_array)
    sorted_ind = [(n//W, n % W) for n in sorted_vote_array]

    for idx in sorted_ind:
        y, x = idx
        max_around = np.max(vote_table[max(0, y-1):min(H, y+1), max(0, x-1):min(W, x+1)])
        if vote_table[y, x] >= max_around:
            N -= 1
            out[y, x] = vote_table[y, x]
        if N == 11:
            break

    return out


def main():
    img_in = cv2.imread("../img/in/thorino.jpg").astype(np.float64).copy()
    vote_table = hough_step1(img_in)
    out = hough_step2(vote_table)
    out_clip = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_45.jpg", out_clip)


if __name__ == '__main__':
    main()
