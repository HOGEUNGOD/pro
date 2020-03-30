import numpy as np
import cv2
import os
import matplotlib as plt

path = 'data/img/'
im_first = 'ff000003.jpg'
im_last = 'ff000056.jpg'
save_path = 'models/'

file_list = os.listdir(path)
count = 0
im1 = cv2.imread(path+ im_first, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(os.path.join(path + im_last), cv2.IMREAD_GRAYSCALE)
im_calculate_first = np.array(im1)
im_calculate = np.array(im2)

np.savetxt('a.txt',im1)