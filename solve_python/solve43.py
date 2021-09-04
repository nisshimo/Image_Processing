import cv2
import numpy as np
from solve42 import canny_step2


# Q_43
def canny(img, ht=50, lt=40):
    """ヒステリシス閾値処理
    """
    H, W, C = img.shape

    edge, angle = canny_step2(img)

    for x in range(W):
        for y in range(H):
            if edge[y, x] >= ht:
                edge[y, x] = 255
            elif edge[y, x] < lt:
                edge[y, x] = 0
            else:
                if edge[y, x] < np.max(edge[max(y-1, 0):min(y+1, H), max(x-1, 0):min(x+1, W)]):
                    edge[y, x] = 255
    return edge, angle


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    edge, angle = canny(img_in)
    edge_out = np.clip(edge, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_43.jpg", edge_out)


if __name__ == '__main__':
    main()
