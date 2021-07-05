from scipy.spatial import distance
import numpy as np

def make_line(coordinates_points):
    result= np.where((distance.cdist(coordinates_points,coordinates_points)<=1) &(distance.cdist(coordinates_points,coordinates_points)>0))
    return np.c_[result]+1

coordinates_points = [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,1),(0.5,0.5,0.5)] ##BCC
# coordinates_points = [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,1),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),(0.5,1,0.5),(1,0.5,0.5),(0.5,0.5,1),(0.5,0.5,0.5)] ## Feb
# coordinates_points = [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,1),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),(0.5,1,0.5),(1,0.5,0.5),(0.5,0.5,1)] ## FCC

print(make_line(coordinates_points))