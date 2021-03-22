import sys
import random
import numpy as np
import cv2


src = cv2.imread(r'E:\experiment data\aluminum\2021-03-19\2_1resoltech\analysis\img\ff0001.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

contours, hier = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

idx = 150
idx = hier[0, idx, 0]
c = (255,255,255)
cv2.drawContours(dst, contours, idx, c, 2, cv2.LINE_8, hier)


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
