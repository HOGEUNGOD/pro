import matplotlib.pyplot as plt
import numpy as np


theta = np.arange(0, 2*np.pi, .01)[1:]

von mises criterion
ax = plt.polar(theta, 1+np.cos(theta)+1.5*np.sin(theta)**2*0.00112311187)



plt.show()
