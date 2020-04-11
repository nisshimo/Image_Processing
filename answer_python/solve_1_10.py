import cv2
import numpy as np


def visualize(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


# Q_1
def swap_channel(img):
    """Swap RGB in order of BGR
    """
    img_ = img[:, :, (2, 1, 0)]
    return img_


# Q_2
def gray_scale(img):
    """Grayscale an image
    """
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    img_gray = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel
    return img_gray


# Q_3
def binarize(img):
    """Binarize image
    """
    img_gray = gray_scale(img)
    img_b = np.where(img_gray < 128, 0, 255)
    return img_b


# Q_4
def binarize_otsu(img):
    """Binarize image with otsu's method
    """
    img_gray = gray_scale(img)
    pixels = img_gray.ravel()

    sigma_b_max = 0
    th = 0
    for i in range(256):
        black = pixels[pixels >= i]
        white = pixels[pixels < i]
        n_b = len(black)
        n_w = len(white)
        m_b = np.mean(black)
        m_w = np.mean(white)

        # variance between class
        sigma_b = n_b * n_w / ((n_b + n_w) ** 2) * ((m_b - m_w) ** 2)
        if sigma_b > sigma_b_max:
            th = i
            sigma_b_max = sigma_b

    img_b = np.where(img_gray < th, 0, 255)
    return img_b


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


# Q_6
def reduce_color(img):
    """Color reduction processing
    """
    return (img // 64) * 64 + 32


# Q_7, Q_8
def pooling(img, stride=8, method=np.max):
    """Average pooling
    """
    out = img.copy()
    for i in range(img.shape[0] // stride):
        for j in range(img.shape[1] // stride):
            h = i*stride
            v = j*stride
            h_ = min((i+1)*stride, img.shape[0])
            v_ = min((j+1)*stride, img.shape[1])
            represent = method(method(img[h:h_, v:v_, :], axis=1), axis=0).reshape(1, 1, 3)
            out[h:h_, v:v_, :] = np.tile(represent, (h_ - h, v_ - v, 1))
    return out


# Q_9
def gaussian_filter(img, stride=3, sigma=1.3):
    K = np.zeros((stride, stride))
    stride_range = range(int(-stride/2), int(stride/2+1))
    for x in stride_range:
        for y in stride_range:
            K[x+1][y+1] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    K /= np.sum(K)

    out = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                subregion = np.zeros_like(K, dtype=np.float64)
                for x in stride_range:
                    for y in stride_range:
                        if 0 <= i + x < img.shape[0] and 0 <= j + y < img.shape[1]:
                            subregion[x+1][y+1] = img[i+x, j+y, k]
                out[i, j, k] = np.sum(subregion * K)
    return out


# Q_10
def median_filter(img, stride=3):
    stride_range = range(int(-stride/2), int(stride/2+1))
    out = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                subregion = np.zeros((stride, stride))
                for x in stride_range:
                    for y in stride_range:
                        if 0 <= i + x < img.shape[0] and 0 <= j + y < img.shape[1]:
                            subregion[x+1][y+1] = img[i+x, j+y, k]
                out[i, j, k] = np.median(subregion)
    return out

def main():
    img = cv2.imread("../imori_noise.jpg").astype(np.float64)
    img_ = median_filter(img)
    img_converted = np.clip(img_, 0, 255).astype(np.uint8)
    visualize(img_converted)


if __name__ == '__main__':
    main()
