"""
Q1 チャネル入れ替え:
    画像を読み込み、RGBをBGRの順に入れ替えよ
"""

import cv2
import numpy as np

img = cv2.imread("imori.jpg")

img2 = img.copy()
img2[:, :] = img[:, :, (2, 1, 0)]

# 結果の出力
cv2.imwrite("ans_1.jpg", img2)
