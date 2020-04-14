import cv2
import numpy as np
from solve32 import dft, idft


# Q_33
def bandpass_filter(G, r_high=2, r_low=0):
    H, W, C = G.shape

    x = np.tile(np.arange(W), (H, 1))
    y = np.repeat(np.arange(H), W).reshape(H, -1)

    idx_x = np.arange(W // 2, W // 2 + W) % W
    idx_y = np.arange(H // 2, H // 2 + H) % H

    x_ = x[:, idx_x].copy()
    y_ = y[idx_y].copy()

    G_ = G[x_, y_]

    d = np.sqrt((x - W//2) ** 2 + (y - H//2) ** 2)

    G_filtered = np.zeros_like(G, dtype=np.complex)

    for c in range(C):
        G_filtered[..., c] = np.where((d >= r_low * min(H, W) // 2) & (d <= r_high * min(H, W) // 2), G_[..., c], 0)
    
    G_out = G_filtered[x_, y_]

    return G_out


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    G = dft(img_in)
    G_ = bandpass_filter(G, r_high=0.5)
    img_ = idft(G_)
    img_out = np.clip(img_, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_33.jpg", img_out)


if __name__ == '__main__':
    main()
