"""
Q3 ニ値化:
    画像を二値化せよ。 二値化とは、画像を黒と白の二値で表現する方法である。 ここでは、グレースケールにおいて閾値を128に設定し、下式で二値化する。
    y = { 0 (if y < 128)
     255 (else) 
"""

import cv2
import numpy as np

img = cv2.imread("imori.jpg")

img_gray = img[:,:,0]*0.0722 + img[:,:,1]*0.7162 + img[:,:,2]*0.2126

img2 = (img_gray >= 128).astype(np.int)*255



# 結果の出力
cv2.imwrite("ans_3.jpg", img2)


