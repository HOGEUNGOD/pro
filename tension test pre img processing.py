import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import PIL.Image as pil
import os

'''at tension test, useint the for clause and reading reverse order 
  then calculate array average difference'''

path = 'data/img'


file_list = os.listdir(path)
for file in file_list:
    if file.endswith(".jpg"):
        im = pil.open(os.path.join(path, file))

'''
#이미지 배열 처리하는 부분
im = pil.open(path_plus+'ff000001.png')
im2 = pil.open('.png')

im_array_original = np.array(im)
im_array_last = np.array(im2)
img_difference = np.array(im2)-np.array(im)
'''