import matplotlib.pylab as plt
import numpy as np
import pandas as pd

xl = pd.read_excel('data/Stress-ML2.xlsx')
data = np.array(xl)


shape = np.shape(data)
print(shape)
y1 = data[:, 0]
y2 = data[:, 1]
stress = plt.plot(y1, color='b')
ml = plt.twinx()
ml.plot(y2, color='r')

plt.grid()

plt.savefig('models\g.png')
