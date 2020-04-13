import numpy as np
import cv2
import os
import matplotlib.pylab as plt
import pandas as pd
from matplotlib.patches import Polygon

"""setting values"""
path = 'G:/experiment data/aluminum/2020-04-10/6/img'

#part2 , location must be x1<x2, y1<y2

slice_x1 = 67
slice_x2 = 167
slice_y1 = 76
slice_y2 = 852

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
stress_range=5301
y1 = result[0:5301]/255
xl = pd.read_csv(path+'/tension.csv')
data = np.array(xl)
x = np.log(data[:, 4]*125+ np.exp(1))
y2 = data[:, 3]*3000/18*(1 +data[:, 4]*125)

fig, ax = plt.subplots(2,1)
ax0 = ax[0].twinx()
ax[0].set_xlabel('strain')
ax[0].set_ylabel('ML_A.U.')
ax0.set_ylabel('STRESS_N')
line1 = ax[0].plot(x, y1, color='b', label="ML(A.U.")
line2 = ax0.plot(x, y2, color='r', label="STRESS(MPA)")
# ax[0].axvspan(0,stress_range, facecolor='g', alpha = 0.25)
# ax[0].set_xlim([0, np.max(x)])


lines = line1 + line2
labels = [l.get_label() for l in lines]
ax[0].legend(lines, labels, loc=0)
ax[0].grid()


stress = y2[0:stress_range]
ml = y1[0:stress_range]
ax[1].plot(stress, ml)
ax[1].grid()
# ax[1].fill_between(stress,ml,0.22,alpha=0.255)
ax[1].set_xlabel('stress')
ax[1].set_ylabel('ML A.U')



data = {'strain':x, 'stress':y2, 'ml':y1}
df = pd.DataFrame(data)
df.to_csv(path+'/a.csv')
fig.tight_layout()
fig.savefig(path+'1.png', dpi = 700)
fig.show()