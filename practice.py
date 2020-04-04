import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import measure
from scipy import ndimage as ndi

im_last = 'data/input/ff000059.jpg'
im_first = 'data/input/ff000003.jpg'
save_path = 'models/'

im1 = cv2.imread(im_first, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(im_last, cv2.IMREAD_GRAYSCALE)
im_calculate_first = np.array(im1)/255
im_calculate = np.array(im2)/255
img_difference = (im_calculate - im_calculate_first)
filt = np.array([[0, 1, 0], [1, 0, -1], [1, 0, 1]])
ct = ndi.convolve(img_difference, filt)*255

cp = plt.contour(ct, 10)
plt.colorbar()
plt.savefig('contour.png')

#
# # Construct some test data
# x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
#
# # Find contours at a constant value of 0.8
# contours = measure.find_contours(r, 0.89)
#
# # Display the image and plot all contours found
# fig, ax = plt.subplots()
# ax.imshow(r, cmap=plt.cm.gray)
#
# for n, contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#
# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()