import cv2
import numpy as np
import matplotlib.pyplot as plt
from solve02 import RGB2GRAY


# Q_19
def hist_of_pixels(img, ax, bins=255):
    """画素の出現回数をグラフに描画する
    """
    pixels = img.ravel()
    ax.hist(pixels, bins=bins, rwidth=0.8, range=(0, 255))
    return ax


def main():
    img_in = cv2.imread("../img/in/imori_dark.jpg").astype(np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    hist_of_pixels(img_in, ax)
    plt.savefig("../img/out/q_20.jpg")


if __name__ == '__main__':
    main()
