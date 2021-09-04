import cv2
import numpy as np
from solve43 import canny
from solve02 import RGB2GRAY


# Q_43
def hough_step1(img):
    H, W, C = img.shape
    rmax = np.round(np.sqrt(H**2 + W**2)).astype(np.int)

    # edge, angle = canny(img, ht=100, lt=30)
    img_gray = RGB2GRAY(img).reshape(H, W)
    edge = cv2.Canny(img_gray.astype(np.uint8), 30, 100).astype(np.float64)
    out = np.zeros((2*rmax, 180), dtype=np.int)

    for x in range(W):
        for y in range(H):
            if edge[y, x] > 0:
                for theta in range(180):
                    theta_r = np.deg2rad(theta)
                    rho = x * np.cos(theta_r) + y * np.sin(theta_r)
                    rho = np.int(rho)
                    out[rho + rmax, theta] += 1
    return out


def main():
    img_in = cv2.imread("../img/in/thorino.jpg").astype(np.float64).copy()
    out = hough_step1(img_in)
    edge_out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_44.jpg", edge_out)


if __name__ == '__main__':
    main()
