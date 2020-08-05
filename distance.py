#%%
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt

read = pd.read_excel('data\distance_angle\input_distance.xlsx')
im =np.array(cv2.imread('data/distance_angle/ff000057.jpg', cv2.IMREAD_GRAYSCALE ))
crack_tip_loaction = np.array([[207,614]])
data = np.array(read)

distance=[]
Ml_intencity=[]


plt.scatter(data[:,3],data[:,2], )

plt.show()

for i in data:
    calculater_location = np.array([[i[2],i[3]]])
    print(calculater_location)
    distance = np.append(distance, cdist(calculater_location, crack_tip_loaction))
    Ml_intencity = np.append(Ml_intencity, np.average(im[int(i[2]):int(i[2]+2),int(i[3]):int(i[3])+2]))




data_union = {'Contour N.o.': data[:,0], 'point': data[:,1],
              'location_y': data[:,2], 'location_x': data[:,3],
              'distance':distance, 'ml_intencity':Ml_intencity}
df = pd.DataFrame(data_union)
df.to_excel('data\distance_angle\output_distance.xlsx')
