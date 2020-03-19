import numpy as np
import matplotlib.pyplot as plt
import math


x = np.linspace(0,90)
a = 0.01*70

list = [i*0.01*37.5/30 for i in range(70,40,-5)]
a = [0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
i=0
for f in list:
    f1=7.12*np.power(f,4)-5.47*np.power(f,3)+2.79*np.power(f,2)+2.63*f+0.20
    y=np.cos(np.deg2rad(x))*np.cos(np.deg2rad(0))*f1*37.5/30
    la=a[i]
    print(f)
    i = i + 1
    plt.plot(x,y,label=la)

plt.xlabel('degree')
plt.ylabel('f1')
plt.legend()
plt.xlim(0,90)
plt.grid()



plt.savefig('models\graph.png', dpi=360)
plt.show()