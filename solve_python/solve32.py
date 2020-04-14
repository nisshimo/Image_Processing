import cv2
import numpy as np
from solve02 import RGB2GRAY


# Q_32
def dft(img):
    H, W, C = img.shape
    root_HW = np.sqrt(H * W)

    G = np.zeros_like(img, dtype=np.complex)
    
    x = np.tile(np.arange(W), (H, 1))
    y = np.repeat(np.arange(H), W).reshape(H, -1)
    
    for c in range(C):
        for k in range(W):
            for l in range(H):
                G[k, l, c] = np.sum(img[..., c] * np.exp(-2 * np.pi * 1j * (k * x / W + l * y / H))) / root_HW
    return G


def idft(G):
    H, W, C = G.shape
    root_HW = np.sqrt(H * W)

    img = np.zeros_like(G, dtype=np.float64)
    
    k = np.tile(np.arange(W), (H, 1))
    l = np.repeat(np.arange(H), W).reshape(H, -1)

    for c in range(C):
        for x in range(W):
            for y in range(H):
                img[x, y, c] = np.abs(np.sum(G[..., c] * np.exp(2 * np.pi * 1j * (k * x / W + l * y / H)))) / root_HW
    return img


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64).copy()
    G = dft(img_in)
    G_scale = np.abs(G) * 255 / np.max(np.abs(G)).copy()
    cv2.imwrite("../img/out/q_32_1.jpg", G_scale)

    img_out = idft(G)
    img_out_clip = np.clip(img_out, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_32_2.jpg", img_out_clip)


if __name__ == '__main__':
    main()
