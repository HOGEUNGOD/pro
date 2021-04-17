import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd
import fracture


#csv파일 형식 바꾸기!

"""setting values"""
path = r'E:\experiment data\aluminum\2021-03-25\faruv_tension_2\img'
path_save= r'E:\experiment data\aluminum\2021-03-25\faruv_tension_2'
# tension_section = 18
# gauge_length = 25
#

#part2 , location must be x1<x2, y1<y2

slice_x1 = 71
slice_x2 = 142
slice_y1 = 414
slice_y2 = 661

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
        print(np.shape(im_calculate), count, ave)
ml_a = result
ml_avg = ml_a
#%%
ml_avg = ml_a - result[0]

#%%
from scipy.signal import savgol_filter
tension_data = np.array(pd.read_csv(path+'/tension.csv', encoding='CP949'))

time, strain, stress = tension_data[:, 0], tension_data[:, 4]*10/25, tension_data[:, 3]*3000/18
ml_k = ml_a[:11750] - result[0]
time = time[:11750]
stress = stress[:11750]
strain = strain[:11750]
stress = savgol_filter(stress,201,1)
strain = savgol_filter(strain,201,1)
ml_avg = savgol_filter(ml_k,6001,1)
# stress, strain=fracture.true_ss(stress,strain)

#############
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.set_xlabel('true strain')
ax.set_ylabel('ml')
ax1.set_ylabel('true stress')
line1 = ax.plot(strain, ml_avg, color='b', label="ml")
line2 = ax1.plot(strain, stress, color='r', label="true stress")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=4)
ax.grid()
fig.savefig(path_save+'/graph.png', dpi = 300)
plt.show()
#
fig, ax2 = plt.subplots()
ax2.plot(stress, ml_avg,c ='black')
ax2.grid()
ax2.set_xlabel('true stress')
ax2.set_ylabel('ML A.U')
fig.savefig(path_save+'/stress_ml.png', dpi = 300)
plt.show()
# fig, ax3 = plt.subplots()
# ax3.plot(strain[:9438], ml_avg[:9438],c = 'black')
# ax3.grid()
# ax3.set_xlabel('displacemnets')
# ax3.set_ylabel('ML A.U')
# fig.savefig(path_save+'/strain_ml.png', dpi = 300)



#%%
b = np.hstack((np.hstack((strain.reshape(-1,1),stress.reshape(-1,1))),ml_avg.reshape(-1,1)))
np.savetxt(path_save+'/stress_ml.csv',b)
print('hi')
#%%
print(strain[-1])
