import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import sys
import glob
from scipy.spatial.distance import cdist

path = 'E:\experiment data\\aluminum\\2021-01-18\\acri_0.05mm_nofilter\img'
save_path = os.path.join(path, "a")
img = glob.glob(path+'/f*.jpg')
if not img:
    print("Image read false!")
    sys.exit()

num=1250
count = 1

src_ = cv2.imread(img[num], cv2.IMREAD_GRAYSCALE)
src1 = src_.copy()
src_0 = cv2.imread(img[750], cv2.IMREAD_GRAYSCALE)
dst = cv2.subtract(src_, src_0)
"""표준편차 양 바꾸면서 그래프 그리기 3*3갯수로 만들기"""

# fig, ax = plt.subplots(3,3, figsize=(15,15))
# for i in range(0,3):
#     for j in range(0,3):
#         gas_sigma = i*10+j*5+1
#         gas = cv2.GaussianBlur(dst,(0,0), gas_sigma)
#         ax[i][j].imshow(src1,cmap='gray')
#         cs = ax[i][j].contour(gas, [k for k in range(2,int(np.max(gas)))],  colors='black', linewidths = 0.5)
#         ax[i][j].set_title(str(gas_sigma)+'gas sigma, level max'+str(int(np.max(gas))))

gas = cv2.GaussianBlur(dst,(0,0), 2)
# plt.imshow(src1,cmap='gray')
# cs = plt.contour(gas, [k for k in range(2,int(np.max(gas)))],  colors='black', linewidths = 0.5)
# cs = plt.contour(gas, np.max(gas),  colors='black', linewidths = 0.5)
# plt.savefig(save_path+'/%d.png'%num)
# print(gas.max(),'gas', dst.max(),'dst')
count += 1


x = src_0[498,80:536]
plt.plot(x)
plt.ylim(35,60)
plt.show()