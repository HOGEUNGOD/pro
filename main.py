import numpy as np
import DIC.DIC as DIC

class G():
    def __init__(self, K1, K2, F_STRESS, r, theta):
        self.K1 = K1
        self.K2 = K2
        self.F_STRESS = F_STRESS
        self.r = r
        self.theta = np.deg2rad(theta)

    def g(self, effective_strain):
        result = 1/(2*np.pi*self.r) * ((self.K1*np.sin(self.theta)+ 2*self.K2*np.cos(self.theta))**2 +(self.K2*np.sin(self.theta)**2)) \
                 + 2 * self.F_STRESS / (np.sqrt(2*np.pi*self.r))*(self.K1*np.sin(1+2*np.cos(self.theta)) + self.K2*(1+2*np.cos(self.theta)**2+np.cos(self.theta))) \
                 + self.F_STRESS**2 - effective_strain
        return result

    def delta_k1(self):
        result = (self.K1*np.sin(self.theta)+2*self.K2*np.cos(self.theta)) / (np.pi * self.r) \
                 + (np.sqrt(2)*self.F_STRESS*(2*np.cos(self.theta)+1)*np.sin(self.theta/2)*np.sin(self.theta))/np.sqrt(np.pi*self.r)
        return result

    def delta_k2(self):
        result = (2*self.K2*np.sin(self.theta)**2 + 4*(self.K1*np.sin(self.theta)+ 2*self.K2*np.cos(self.theta))*np.cos(self.theta))/(2*np.pi*self.r) \
                 + (np.sqrt(2)*self.F_STRESS*(2*np.cos(self.theta)**2 + np.cos(self.theta)+1)*np.sin(self.theta/2))/np.sqrt(np.pi*self.r)
        return result

    def delta_stress(self):
        result = 2*self.F_STRESS + ( np.sqrt(2)* np.sin(self.theta/2)* (self.K1 * (2*np.cos(self.theta)+1)*np.sin(self.theta) + self.K2*(2*np.cos(self.theta)**2 +np.cos(self.theta) +1)))/np.sqrt(np.pi*self.r)
        return result

iteration = 2
# youngs_modulus =
# poisson_ratio =
# k =


K1, K2, F_STRESS= 1,1,1
gk_list=[]
location = [[1,2,3,60],[5,4,3,80]]
A_matrix = [0,0,0]
G_martix = []

for i in range(iteration):
    for _, _, degree, rho in location:
        gk = G(K1,K2,F_STRESS,rho, degree)
        A_matrix = np.vstack((A_matrix,[gk.delta_k1(), gk.delta_k2(), gk.delta_stress()]))
        G_martix.append(gk.g(0.5))
    A_matrix = - A_matrix[1:,]
    G_martix = np.reshape(G_martix,(-1,1))
    delta_matrix = np.dot(np.linalg.inv(np.dot(A_matrix.transpose(), A_matrix)),np.dot(A_matrix.transpose(),G_martix))
    gk_list.append(np.average(G_martix))
    K1 = K1+delta_matrix[0]
    K2 = K2+delta_matrix[1]
    F_STRESS = F_STRESS + delta_matrix[2]
    print(A_matrix)

