import numpy as np
import cv2
import os
import matplotlib as plt

path = 'data/img/'
im_first = 'ff000003.jpg'
save_path = 'models/'

file_list = os.listdir(path)
count = 0
for file in file_list:
    if file.endswith(".jpg"):
        im1 = cv2.imread(path+ im_first, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        im_calculate_first = np.array(im1)
        im_calculate = np.array(im2)
        img_difference = im_calculate - im_calculate_first
        cv2.imwrite(save_path+file, img_difference)

        count += 1

