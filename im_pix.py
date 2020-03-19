import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt


# Read image
im = pilimg.open('ff000001.png')


# Fetch image pixel data to numpy array
pix = np.array(im)
cmax = np.max(pix)
print('최대값은 : {}'.format(cmax))
cmax_index = np.where(cmax == pix)
print ('최대값 {}의 index는 {}'.format(cmax, cmax_index))

np.where(max)
plt.imshow(pix, cmap='gray')
plt.show()