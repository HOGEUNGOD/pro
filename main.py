import numpy as np
import fuction
import pandas as pd
import matplotlib.pyplot as plt

path = 'G:/experiment data/aluminum/2020-04-17/1/img'

tension_data = np.array(pd.read_csv(path+'/tension.csv', encoding='CP949'))
time, strain, stress = tension_data[:, 0], tension_data[:, 1]/20, tension_data[:, 2]/18

true = fuction.true_ss(stress, strain)
true_stress, true_strain = true[0], true[1]
aaa= np.resize(true_stress, (np.shape(true_stress)[0]//2, 2))


data = {'time':time, 'strain':strain, 'stress':stress,
        'true_strain':true_strain, 'true_stress':true_stress}
df = pd.DataFrame(data)
df.to_csv(path+'/stress.csv')

plt.plot(strain, stress)
plt.savefig(path+'/aad.png')
plt.show()
# true = fuction.true_ss(stress, strain)