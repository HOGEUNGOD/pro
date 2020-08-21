import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd
import fuction

#csv파일 형식 바꾸기!

"""setting values"""
path = 'G:/experiment data/aluminum/2020-04-17/1/img'
path_save= 'G:/experiment data/aluminum/2020-04-17/1'
tension_section = 18
gauge_length = 20
time_proportion = 6
volt_startframe = 110

#part2 , location must be x1<x2, y1<y2

slice_x1 = 129
slice_x2 = 200
slice_y1 = 335
slice_y2 = 662

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
        print(np.shape(im_calculate), count)

ml_avg = result - result[0]
tension_data = np.array(pd.read_csv(path+'/tension.csv', encoding='CP949'))
time, strain, stress = tension_data[:, 0], tension_data[:, 1]/gauge_length, tension_data[:, 2]/tension_section
true = fuction.true_ss(stress, strain)
frame_endpoint = np.shape(time)[0]
true_stress, true_strain = true[0], true[1]
ml_A = np.resize(ml_avg[volt_startframe:], (np.shape(ml_avg[volt_startframe:])[0]//time_proportion+1, time_proportion))[0:frame_endpoint, 0]


data = {'time': time, 'strain': strain, 'stress': stress,
        'true_strain': true_strain, 'true_stress': true_stress,
        'ML': ml_A, 'start file num': volt_startframe}
df = pd.DataFrame(data)
df.to_csv(path_save+'/data.csv')



#graph part
#############
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.set_xlabel('TRUE STRAIN')
ax.set_ylabel('ML_A.U.')
ax1.set_ylabel('TRUE STRESS(MAP)')
line1 = ax.plot(true_strain, true_stress, color='b', label="ML(A.U.)")
line2 = ax1.plot(true_strain, ml_A, color='r', label="TRUE STRESS")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=0)
ax.grid()
fig.savefig(path_save+'/graph.png', dpi = 1000)

fig, ax2 = plt.subplots()
ax2.plot(true_stress, ml_A)
ax2.grid()
ax2.set_xlabel('true stress')
ax2.set_ylabel('ML A.U')
fig.savefig(path_save+'/graph1.png', dpi = 1000)
##############

fig, ax = plt.subplots(2, 1)
ax0 = ax[0].twinx()
ax[0].set_xlabel('TRUE STRAIN')
ax[0].set_ylabel('ML_A.U.')
ax0.set_ylabel('TRUE STRESS(MAP)')
line1 = ax[0].plot(true_strain, true_stress, color='b', label="TRUE STRESS")
line2 = ax0.plot(true_strain, ml_A, color='r', label= "ML(A.U.)")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax[0].legend(lines, labels, loc=0)
ax[0].grid()

ax[1].plot(true_stress, ml_A)
ax[1].grid()
ax[1].set_xlabel('true stress')
ax[1].set_ylabel('ML A.U')

fig.tight_layout()
fig.savefig(path_save+'/graph2.png', dpi = 1000)
fig.show()