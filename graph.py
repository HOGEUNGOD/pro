import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0.001, 1)

y1 = x
y2 = np.sqrt((8/np.power(np.pi, 2))*np.log(1/np.cos(np.pi*x/2)))
y3 = x*np.sqrt(1+0.5*np.power(x,2))

fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

line1 = ax1.plot(x, y1, color='b', label="from LEFM", linestyle ='-')
line2 = ax1.plot(x, y2, color='r', label="strip yield correction", linestyle ='--')
line3 = ax1.plot(x, y3, color='g', label="from Irwin correction for plane stress", linestyle='-.')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
plt.xlim(-0.1, 1.2)
plt.grid()
plt.legend(lines, labels, loc=0)
plt.savefig('models\graph.png', dpi=600)
plt.show()