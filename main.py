
import mecha.DIC as dic
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mecha.fracture as fracture
from scipy import io
import mecha.DIC as dic
import matplotlib.patches as patches

j = 4


image = dic.load_file('../data/CTS')
left, right, up_down = 50,100,75
crack_tip = [[400,207],[397,210],[389,205],[387,202],[386,200]] # y, x
contour_level = 17
ref = list(image.keys())[0]
cur = list(image.keys())[j]
src_reference = image[ref][:,:,1][crack_tip[0][0]-up_down:crack_tip[0][0]+up_down,crack_tip[0][1]-left:crack_tip[0][1]+right]
src_current = image[cur][:,:,1][crack_tip[j][0]-up_down:crack_tip[j][0]+up_down,crack_tip[j][1]-left:crack_tip[j][1]+right]

dst = cv2.subtract(src_current.astype('float64'),src_reference.astype('float64')).astype(float)
dst = np.where(dst>10, dst*0, dst)
dst = np.where(dst<0, dst*0, dst)
gas = cv2.GaussianBlur(dst,(0,0),10)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(src_current, cmap='gray')
# plt.axis('off')
csf = ax.contourf(gas/150, contour_level, cmap = 'jet')
CS = ax.contour(gas/150, contour_level, colors='black', linewidths = 0.5)
cbar= plt.colorbar(csf)
cbar.ax.tick_params(labelsize=18)
ax.add_patch(
     patches.Rectangle(
        (105, 134),
        30,
        2,
        edgecolor = 'black',
        facecolor = 'black',
        fill=True
     ) )
plt.text( 113,143, '1mm', fontsize=15, fontweight='bold')
plt.axis('off')
plt.savefig('../data/CTS/{}_contour_나누기.png'.format(str(j)),dpi=300,bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(src_current, cmap='gray')
# plt.axis('off')
csf = ax.contourf(gas, contour_level, cmap = 'jet')
CS = ax.contour(gas, contour_level, colors='black', linewidths = 0.5)
cbar= plt.colorbar(csf)
cbar.ax.tick_params(labelsize=18)
ax.add_patch(
     patches.Rectangle(
        (105, 134),
        30,
        2,
        edgecolor = 'black',
        facecolor = 'black',
        fill=True
     ) )
plt.text( 113,143, '1mm', fontsize=15, fontweight='bold')
plt.axis('off')

plt.savefig('../data/CTS/{}_contour.png'.format(str(j)),dpi=300,bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(src_current, cmap='gray')
ax.add_patch(
     patches.Rectangle(
        (105, 134),
        30,
        2,
        edgecolor = 'black',
        facecolor = 'black',
        fill=True
     ) )
plt.text( 113,143, '1mm', fontsize=15, fontweight='bold')
plt.axis('off')
plt.savefig('../data/CTS/{}_raw.png'.format(str(j)),dpi=300,bbox_inches='tight')
plt.show()
plt.close()

g_frame= np.where(src_current>40, 0,255)
h, w = g_frame.shape
img = np.ones([h, w, 4], np.uint8) * (0, 0, 0, 0)
img[:,:,3] = g_frame
cv2.imwrite('../data/CTS/{}_mask.png'.format(str(j)), img)


uu = io.loadmat('../data/CTS/data'+'/'+str(j)+'_uu.mat')['uu']
vv = io.loadmat('../data/CTS/data'+'/'+str(j)+'_vv.mat')['vv']
exx = io.loadmat('../data/CTS/data'+'/'+str(j)+'_exx.mat')['exx']
eyy = io.loadmat('../data/CTS/data'+'/'+str(j)+'_eyy.mat')['eyy']
exy = io.loadmat('../data/CTS/data'+'/'+str(j)+'_exy.mat')['exy']
ee = fracture.effective_strain(exx, eyy, exy)
uu_resize = cv2.resize(uu,(0,0),fx=2, fy=2 )
vv_resize = cv2.resize(vv,(0,0),fx=2, fy=2 )
exx_resize = cv2.resize(exx,(0,0),fx=2, fy=2 )
eyy_resize = cv2.resize(eyy,(0,0),fx=2, fy=2 )
exy_resize = cv2.resize(exy,(0,0),fx=2, fy=2 )
ee_resize = cv2.resize(ee,(0,0),fx=2, fy=2 )

ee_dic = ee_resize[crack_tip[j][0]-up_down:crack_tip[j][0]+up_down,crack_tip[j][1]-left:crack_tip[j][1]+right]
ee_dic = ee_dic.astype(float)
ee_dic = np.where(ee_dic>10, ee_dic*0, ee_dic)


fig, ax = plt.subplots(figsize=(10, 10))
ax.add_patch(
     patches.Rectangle(
        (105, 134),
        30,
        2,
        edgecolor = 'black',
        facecolor = 'black',
        fill=True
     ) )
plt.text( 113,143, '1mm', fontsize=15, fontweight='bold')
ax.imshow(ee_dic, cmap='gray')
csf = ax.contourf(ee_dic, contour_level, cmap = 'jet')
CS = ax.contour(ee_dic, contour_level, colors='black', linewidths = 0.5)
cbar= plt.colorbar(csf)
cbar.ax.tick_params(labelsize=18)
plt.axis('off')
plt.savefig('../data/CTS/{}_dic_ee.png'.format(str(j)),dpi=300,bbox_inches='tight')
plt.show()
plt.close()

