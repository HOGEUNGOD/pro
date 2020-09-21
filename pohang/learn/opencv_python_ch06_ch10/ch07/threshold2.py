import sys
import numpy as np
import cv2


src = cv2.imread('rice.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()


_, dst1 = cv2.threshold(src,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dst2 = np.zeros(src.shape, np.uint8)


cv2.imshow('src', src)
cv2.namedWindow('dst')


cv2.waitKey()
cv2.destroyAllWindows()
