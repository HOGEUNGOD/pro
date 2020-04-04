import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import PIL.Image as pil

path = 'data/composite/'
#part1
fisrt_img = 'ff000001.jpg'
last_img = 'ff000423.jpg'

################part1##############################
im = cv2.imread(path+fisrt_img, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(path+last_img, cv2.IMREAD_GRAYSCALE)
im_array_original = np.array(im)
im_array_last = np.array(im2)
img_difference = np.array(im2)-np.array(im)

cv2.imwrite("models/i.jpg", img_difference)