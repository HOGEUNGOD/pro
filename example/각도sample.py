import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(projection = "polar")

r = [0, 100]
theta_max = [0, np.pi*2/3]
theta_min = [0, np.pi*4/3]

result = []
for i in range(1,9):
    k = (np.pi*4/3 - np.pi*2/3)*i/9 + np.pi*2/3
    result = np.append(result, [k])


ax.plot(theta_min, r, color="red")
ax.plot(theta_max, r, color="red")
ax.axis("off")

for i in result:
    ax.plot([0,i], r, color="yellow", linewidth= "0.5")



ax.tick_params(grid_color= "white")
plt.savefig('models/bb.png',format='png', transparent=True)
plt.show()