import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd
from matplotlib.patches import Polygon

"""setting values"""
path = 'G:/experiment data/aluminum/2020-04-03/4/img'

#part2 , location must be x1<x2, y1<y2

slice_x1 = 74
slice_x2 = 185
slice_y1 = 146
slice_y2 = 859

file_list = os.listdir(path)
count = 0
result = []

for file in file_list:
    if file.endswith(".png"):
        im = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        im_calculate = np.array(im)
        img_box = im_calculate[slice_y1:slice_y2, slice_x1:slice_x2]
        ave = np.average(img_box)
        result = np.append(result, [ave])
        count += 1
        print(np.shape(im_calculate), count)


#graph part
stress_range=18000
y1 = result/255
xl = pd.read_csv(path+'/tension.csv')
data = np.array(xl)
x = data[:, 1]
y2 = data[:, 3]*3000/18

fig, ax = plt.subplots(2,1)
ax0 = ax[0].twinx()
ax[0].set_xlabel('frame N.o.')
ax[0].set_ylabel('ML_A.U.')
ax0.set_ylabel('STRESS_N')
line1 = ax[0].plot(x, y1, color='b', label="ML(A.U.")
line2 = ax0.plot(x, y2, color='r', label="STRESS(MPA)")
ax[0].axvspan(0,stress_range, facecolor='g', alpha = 0.25)
ax[0].set_xlim([0, np.max(x)])

data = {'stress':y2, 'ml':y1}
df = pd.DataFrame(data)
df.to_csv(path)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax[0].legend(lines, labels, loc=0)
ax[0].grid()


stress = y2[0:stress_range]
ml = y1[0:stress_range]
ax[1].plot(stress, ml)
ax[1].grid()
ax[1].fill_between(stress,ml,0.22,alpha=0.255)
ax[1].set_xlabel('stress')
ax[1].set_ylabel('ML A.U')

fig.tight_layout()
fig.savefig(path+'1.png', dpi = 700)
fig.show()