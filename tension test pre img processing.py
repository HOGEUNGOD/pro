import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd
import fracture


#csv파일 형식 바꾸기!

"""setting values"""
path = r'E:\experiment data\aluminum\2021-03-25\faruv_tension_2\img'
path_save= r'E:\experiment data\aluminum\2021-03-25\faruv_tension_2\ML_STRESS'
tension_section = 18
gauge_length = 25


#part2 , location must be x1<x2, y1<y2
slice_x1 = 96
slice_x2 = 110
slice_y1 = 500
slice_y2 = 633

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
#%%

ml_a = result

tension_data = np.array(pd.read_csv(path+'/tension.csv', encoding='CP949'))
time, strain, stress = tension_data[:, 0], tension_data[:, 4]*20, tension_data[:, 3]*3000
# true = fracture.true_ss(stress, strain)
# frame_endpoint = np.shape(time)[0]
# true_stress, true_strain = true[0], true[1]


#
# data = {'time': time, 'strain': strain, 'stress': stress,
#         'true_strain': true_strain, 'true_stress': true_stress,
#         'ML': ml_a,}
# df = pd.DataFrame(data)
# df.to_csv(path_save+'/data.csv')
#
# #%%
# from scipy.signal import savgol_filter
# # ml_avg = savgol_filter(ml_a[:9438],2001,1)
# ml_avg = ml_a[:9438]
# #graph part
#############
ml_avg = ml_a

fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.set_xlabel('time')
ax.set_ylabel('Stress')
ax1.set_ylabel('ml_avg A.U.')
line1 = ax.plot(time, stress, color='b', label="STRESS")
line2 = ax1.plot(time, ml_avg, color='r', label="ML(A.U.)")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.set_xlim(0,16700)
ax1.set_ylim(0.34, 0.425)
ax.legend(lines, labels, loc=0)
ax.grid()
fig.savefig(path_save+'/graph.png', dpi = 300)
plt.show()


fig, ax2 = plt.subplots()
ax2.plot(stress, ml_avg,c ='black')
ax2.grid()
ax2.set_xlabel('stress')
ax2.set_ylabel('ML A.U')
fig.savefig(path_save+'/stress_ml.png', dpi = 300)

fig, ax3 = plt.subplots()
ax3.plot(strain, ml_avg,c = 'black')
ax3.grid()
ax3.set_xlabel('displacemnets')
ax3.set_ylabel('ML A.U')
fig.savefig(path_save+'/strain_ml.png', dpi = 300)

