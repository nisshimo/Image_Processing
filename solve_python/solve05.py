import cv2
import numpy as np


# Q_5
def RGB2HSV(img):
    """convert Convert HSV into RGB pixels
    """
    img = img / 255
    B = img[..., 0].copy()
    G = img[..., 1].copy()
    R = img[..., 2].copy()
    Max = np.max(img, axis=2)
    Min = np.min(img, axis=2)

    H = np.zeros(B.shape)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if Min[i][j] == Max[i][j]:
                H[i][j] = 0
            elif Min[i][j] == B[i][j]:
                H[i][j] = 60 * (G[i][j] - R[i][j]) / (Max[i][j] - Min[i][j]) + 60
            elif Min[i][j] == R[i][j]:
                H[i][j] = 60 * (B[i][j] - G[i][j]) / (Max[i][j] - Min[i][j]) + 180
            elif Min[i][j] == G[i][j]:
                H[i][j] = 60 * (R[i][j] - B[i][j]) / (Max[i][j] - Min[i][j]) + 300
    S = Max - Min
    V = Max
    return H, S, V


def HSV2RGB(H, S, V):
    """convert Convert RGB pixels into HSV
    """
    C = S
    H_ = H / 60
    X = C * (1 - np.abs(H_ % 2 - 1))

    img_rgb = np.tile((V - C).reshape(C.shape[0], C.shape[1], 1), (1, 1, 3))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            h = H_[i][j]
            if 0 <= h < 1:
                img_rgb[i, j, :] += np.array([0, X[i][j], C[i][j]])
            elif 1 <= h < 2:
                img_rgb[i, j, :] += np.array([0, C[i][j], X[i][j]])
            elif 2 <= h < 3:
                img_rgb[i, j, :] += np.array([X[i][j], C[i][j], 0])
            elif 3 <= h < 4:
                img_rgb[i, j, :] += np.array([C[i][j], X[i][j], 0])
            elif 4 <= h < 5:
                img_rgb[i, j, :] += np.array([C[i][j], 0, X[i][j]])
            elif 5 <= h < 6:
                img_rgb[i, j, :] += np.array([X[i][j], 0, C[i][j]])
    return img_rgb*255


def invert_hue(img):
    """Invert the hue of an image
    """
    H, S, V = RGB2HSV(img)
    img_ = HSV2RGB((H + 180) % 360, S, V)
    return img_


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)
    img_ = invert_hue(img_in)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_05.jpg", img_out)

if __name__ == '__main__':
    main()
