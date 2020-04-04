import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd

"""setting values"""
path = 'data/composite/'

#part2 , location must be x1<x2, y1<y2

slice_x1 = 157
slice_x2 = 263
slice_y1 = 45
slice_y2 = 915


file_list = os.listdir(path)
count = 0
result = []

for file in file_list:
    if file.endswith(".jpg"):
        im = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        im_calculate = np.array(im)
        img_box = im_calculate[slice_y1:slice_y2, slice_x1:slice_x2]
        ave = np.average(img_box)
        result = np.append(result, [ave])
        count += 1
        print(np.shape(im_calculate), count)

print(x)

#graph part
y1 = result
xl = pd.read_csv('data/wvd_000_WaveData.csv')
data = np.array(xl)
x = data[:, 0]
y2 = data[:, 3]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

line1 = ax1.plot(x, y1, color='b', label="ML")
line2 = ax2.plot(x, y2, color='r', label="STRESS")

lines = line1 + line2
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc=0)


plt.savefig('models\graph.png', dpi=600)