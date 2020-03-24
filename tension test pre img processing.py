import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import PIL.Image as pil

'''at tension test, useint the for clause and reading reverse order 
  then calculate array average difference'''

path_plus = 'data/'

path = 'Stress-ML2.xlsx'
data = pd.read_excel(path_plus+path)

#이미지 배열 처리하는 부분
im = pil.open(path_plus+'ff000001.png')
im2 = pil.open('.png')

im_array_original = np.array(im)
im_array_last = np.array(im2)
img_difference = np.array(im2)-np.array(im)
