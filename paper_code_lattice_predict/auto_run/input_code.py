
# -*- coding: mbcs -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import numpy as np

structure_name = 'Ferob'
coordinates_points = [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,1),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),(0.5,1,0.5),(1,0.5,0.5),(0.5,0.5,1),(0.5,0.5,0.5)
                      ]

coordinates_lines = [[1,2],[1,3],[1,5],[2,4],[2,6],[3,4],[3,7],[4,8],[5,6],[5,7],[6,8],[7,8],[9,10],[9,11],[9,12],[9,13],[10,11],[10,13],[10,14],[11,12],[11,14],[12,13],[12,14],[13,14]
                     ]
X_num, Y_num, Z_num = 2,2,2

mdb.models['Model-1'].Part(dimensionality=THREE_D, name=structure_name, type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].parts[structure_name].ReferencePoint(point=(0,0,0))

for X in range(X_num):
    for Y in range(Y_num):
        for Z in range(Z_num):
            for points in coordinates_points:
                mdb.models['Model-1'].parts[structure_name].DatumPointByCoordinate(coords=(points[0]+X,points[1]+Y,points[2]+Z))

for i in range(X_num * Y_num * Z_num):
    print(i)
    for lines in coordinates_lines:
        mdb.models['Model-1'].parts[structure_name].WirePolyLine(mergeWire=OFF, meshable=ON,
            points=((mdb.models['Model-1'].parts[structure_name].datums[lines[0]+1+15*i], mdb.models['Model-1'].parts[structure_name].datums[lines[1]+1+15*i])))


print('End')
# Save by ghrms on 2021_03_25-13.54.45; build 6.13-1 2013_05_16-11.28.56 126354
