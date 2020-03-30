import numpy as np
import matplotlib.pyplot as plt

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