import numpy as np
import matplotlib.pyplot as plt
import math


x = np.linspace(0,90)
a = 0.01*70

list = [i*0.01*30/37.5 for i in range(45,75,5)]
a = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
i=0
for f in list:
    f1=2.80*np.power(f,4)-6.5*np.power(f,3)+2.79*np.power(f,2)+2.63*f+0.12
    y=np.sin(np.deg2rad(x))*np.cos(np.deg2rad(0))*f1
    la=a[i]
    i = i + 1
    plt.plot(x,y,label=la)

plt.xlabel('degree')
plt.ylabel('f2')
plt.legend()
plt.xlim(0,90)
plt.grid()



plt.savefig('models\graph2.png', dpi=360)
plt.show()