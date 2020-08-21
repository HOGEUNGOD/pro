import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fuction

xl = pd.read_excel('data/astm.xlsx')
data = np.array(xl)
x = []
y = []
print(np.shape(data))

for i in data:
    nuber = i[0]; a = i[4]*10; p = i[8]/1000; b = 0.3; w = 3
    a=fuction.CTS(a,w,p,b)
    a.k1()
    kk = round(a.k1())
    y.append(kk)
    x.append(a.k1())

plt.plot(x,y)
plt.scatter(x, y, c='red')
plt.xlabel('a/w')
plt.ylabel('elastic k')

plt.grid()
plt.savefig('models\graph2.png',dpi=600)
plt.show()