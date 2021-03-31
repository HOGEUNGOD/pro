import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd
import fracture


#csv파일 형식 바꾸기!

"""setting values"""
path = r'E:\experiment data\aluminum\2021-03-30\3_normal\img'
path_save= r'E:\experiment data\aluminum\2021-03-30\3_normal'
# tension_section = 18
# gauge_length = 25
#

#part2 , location must be x1<x2, y1<y2

slice_x1 = 157
slice_x2 = 476
slice_y1 = 194
slice_y2 = 566

file_list = os.listdir(path)
count = 0
result = []

for file in file_list:
    if file.endswith(".jpg"):
        im = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)/255
        im_calculate = np.array(im)
        img_box = im_calculate[slice_y1:slice_y2, slice_x1:slice_x2]

        ave = np.average(img_box)
        result = np.append(result, [ave])
        count += 1
        print(np.shape(im_calculate), count, ave)
ml_a = result
ml_avg = ml

tension_data = np.array(pd.read_csv(path+'/tension.csv', encoding='CP949'))

time, strain, stress = tension_data[:, 0], tension_data[:, 1], tension_data[:, 2]

#############
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.set_xlabel('time')
ax.set_ylabel('ml_avg A.U.')
ax1.set_ylabel('stress(N)')
line1 = ax.plot(time, ml_avg, color='b', label="ML")
line2 = ax1.plot(time, stress, color='r', label="Stress")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=0)
ax.grid()
fig.savefig(path_save+'/graph.png', dpi = 300)
plt.show()
#
# fig, ax2 = plt.subplots()
# ax2.plot(stress[:9438], ml_avg[:9438],c ='black')
# ax2.grid()
# ax2.set_xlabel('stress')
# ax2.set_ylabel('ML A.U')
# fig.savefig(path_save+'/stress_ml.png', dpi = 300)
#
# fig, ax3 = plt.subplots()
# ax3.plot(strain[:9438], ml_avg[:9438],c = 'black')
# ax3.grid()
# ax3.set_xlabel('displacemnets')
# ax3.set_ylabel('ML A.U')
# fig.savefig(path_save+'/strain_ml.png', dpi = 300)
