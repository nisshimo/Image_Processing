import cv2
import numpy as np
from solve02 import RGB2GRAY


# Q_15
def sobel_filter(img):
    if len(img.shape) == 3:
        assert img.shape[2] == 1
        img = np.squeeze(img, axis=-1)
    H, W = img.shape

    Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
    stride = 3

    pad = stride // 2
    img_pad = np.pad(img, pad_width=pad, mode='constant')

    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float64)

    out_v = out.copy()
    out_h = out.copy()

    for x in range(W):
        for y in range(H):
            out_v[pad + y, pad + x] = np.sum(Kv * img_pad[y:y+stride, x:x+stride])
            out_h[pad + y, pad + x] = np.sum(Kh * img_pad[y:y+stride, x:x+stride])
    out_v = out_v[pad:pad+H, pad:pad+W]
    out_h = out_h[pad:pad+H, pad:pad+W]
    return out_v, out_h


def main():
    img_in = cv2.imread("../img/in/imori.jpg").astype(np.float64)

    img_v, img_h = sobel_filter(RGB2GRAY(img_in))
    img_clip_v = np.clip(img_v, 0, 255).astype(np.uint8)
    img_clip_h = np.clip(img_h, 0, 255).astype(np.uint8)
    cv2.imwrite("../img/out/q_15_v.jpg", img_clip_v)
    cv2.imwrite("../img/out/q_15_h.jpg", img_clip_h)


if __name__ == '__main__':
    main()
