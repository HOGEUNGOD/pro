import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fuction import CTS_Zeinedni

xl = pd.read_excel('data/astm.xlsx')
data = np.array(xl)
x = []
y = []
print(np.shape(data))

for i in data:
    nuber = i[0]; a = i[4]; p = i[8]; b = 10; w = 30
    k = CTS_Zeinedni(a,w,p,b)
    kk = round(k[0])
    y.append(kk)
    x.append(k[1])

plt.plot(x,y)
plt.scatter(x, y, c='red')
plt.xlabel('a/w')
plt.ylabel('elastic k')
plt.ylim(0,2000)
plt.grid()
plt.savefig('models\graph2.png',dpi=600)
plt.show()