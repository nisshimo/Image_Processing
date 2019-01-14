"""
Q2 画像のグレースケール化:
    画像をグレースケールにせよ。 グレースケールとは、画像の輝度表現方法の一種であり下式で計算される。
    Y = 0.2126 R + 0.7152 G + 0.0722 B
"""

import cv2
import numpy as np

img = cv2.imread("imori.jpg")

img2 = img[:,:,0]*0.0722 + img[:,:,1]*0.7162 + img[:,:,2]*0.2126

# 結果の出力
cv2.imwrite("ans_2.jpg", img2)

