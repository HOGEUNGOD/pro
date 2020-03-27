import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd

"""setting values"""
path = 'data/composite/'
#part1
fisrt_img = 'ff000001.jpg'
last_img = 'ff000423.jpg'
#part2 , location_szie must be even
center_location = (424, 562)
location_size_width = 200
location_size_length = 200



################part1##############################
im = cv2.imread(path+fisrt_img, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(path+last_img, cv2.IMREAD_GRAYSCALE)
im_array_original = np.array(im)
im_array_last = np.array(im2)
img_difference = np.array(im2)-np.array(im)

cv2.imwrite("models/i.jpg", img_difference)


#################part2#############################
file_list = os.listdir(path)
count = 0
length = location_size_length/2
width = location_size_width/2
result = []

slice_x1 = int(center_location[1]-width)
slice_x2 = int(center_location[1]+width)
slice_y1 = int(center_location[0]-length)
slice_y2 = int(center_location[0]+length)


for file in file_list:
    if file.endswith(".jpg"):
        im = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        im_calculate = np.array(im)
        img_box = im_calculate[slice_y1:slice_y2, slice_x1:slice_x2]
        ave = np.average(img_box)
        result = np.append(result, [ave])
        count += 1


        print(np.shape(im_calculate), count)

y1 = result

xl = pd.read_csv('data/wvd_000_WaveData.csv')
data = np.array(xl)
x = data[:, 0]
y2 = data[:, 3]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

line1 = ax1.plot(x ,y1, color='b', label="ML")
line2 = ax2.plot(x, y2, color='r', label="STRESS NONE CONVERT")

lines = line1 + line2
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc=0)


plt.savefig('models\graph.png', dpi=600)