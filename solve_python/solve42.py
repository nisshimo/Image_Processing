import cv2
import numpy as np
from solve41 import canny_step1


# Q_42
def canny_step2(img):
    """細線化
    """
    H, W, C = img.shape
    
    edge, angle = canny_step1(img)
    _edge = edge.copy()

    for x in range(W):
        for y in range(H):
            if angle[y, x] == 0:
                dx1, dy1, dx2, dy2 = -1, 0, 1, 0
            elif angle[y, x] == 45:
                dx1, dy1, dx2, dy2 = -1, 1, 1, -1
            elif angle[y, x] == 90:
                dx1, dy1, dx2, dy2 = 0, -1, 0, 1
            elif angle[y, x] == 135:
                dx1, dy1, dx2, dy2 = -1, -1, 1, 1
            if x == 0:
                dx1 = max(dx1, 0)
                dx2 = max(dx2, 0)
            if x == W-1:
                dx1 = min(dx1, 0)
                dx2 = min(dx2, 0)
            if y == 0:
                dy1 = max(dy1, 0)
                dx2 = max(dy2, 0)
            if y == H-1:
                dy1 = min(dy1, 0)
                dy2 = min(dy2, 0)
            if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                _edge[y, x] = 0
    return _edge, angle


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    edge, angle = canny_step2(img_in)
    edge_out = np.clip(edge, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_42.jpg", edge_out)


if __name__ == '__main__':
    main()
