import numpy as np
import math
import matplotlib.pyplot as plt

def ASTM_E399(Pq,a,B,W):
    """B is thickness, W is width, Pq is stress, A is crack length"""

    f_in = a/W
    f=((2+f_in)*(0.886+4.64*f_in-13.32*(f_in**2)+14.72*(f_in**3)-5.6*(f_in**4)))/((1-f_in)**1.5)
    kq=(Pq*f)/(B*(W**0.5))
    return kq

def true_ss(stress, strain):
    """calculate tension test, convert stress, strain to true stress and true strain"""
    true_stress = stress*(1 + strain)
    true_strain = np.log(1 + strain)
    result = np.array([true_stress, true_strain])
    return result

class CTS:
    """Calculate CTS Elastic stress intensity factor"""
    def __init__(self, a, w, p, b):
        self.a = a
        self.w = w
        self.p = p
        self.b = b

    def k1(self):
        #mode 1
        f = self.a / self.w
        f1 = 7.12 * math.pow(f, 4) - 5.47 * math.pow(f, 3) + 2.79 * math.pow(f, 2) + 2.63 * f + 0.20
        k1 = (math.sqrt(self.a * math.pi) * self.p *f1) / (self.w*self.b)
        return k1

    def k2(self):
        f = self.a / self.w
        f2 = 2.8 * math.pow(f, 4) - 6.5 * math.pow(f, 3) + 2.79 * math.pow(f, 2) + 2.63 * f + 0.12
        k2 = (math.sqrt(self.a * math.pi) * self.p *f2) / (self.w*self.b)
        return k2



def K_deviator(sigma, r, theta_degree):
    theta = np.deg2rad(theta_degree)
    K = (sigma * (2 * np.pi * r) ** 0.5)/(np.cos(theta/2)*((1 + 3*np.sin(theta/2) ** 2)**0.5))
    return K

def K_sigma(K,r_mm,theta_degree):
    """Unit K:mpa sqrt(m), r : mm , theta_dgree = dgree"""
    r = r_mm * 0.001
    theta = np.deg2rad(theta_degree)
    sigma = (K * (np.cos(theta/2)*((1 + 3*np.sin(theta/2) ** 2)**0.5))/((2 * np.pi * r) ** 0.5))
    return sigma


class Plastic_zone:
    """input Stress intensity factor, Yeild_strenth """
    def __init__(self, K1, yeild_strenth):
        self.theta = np.arange(0, 2*np.pi, .01)[1:]
        self.K1 = K1
        self.yeild_s = yeild_strenth
        self.von = []
        self.tera = []

    def von_mises(self):
        von_mises = (1+np.cos(self.theta)+1.5*np.sin(self.theta)**2)*(1/(4*np.pi)*(self.K1/self.yeild_s)**2)
        self.von = von_mises
        von_mises_0 = (1 + np.cos(0) + 1.5 * np.sin(0) ** 2) * (1 / (4 * np.pi) * (self.K1 / self.yeild_s) ** 2)
        return von_mises,self.theta, von_mises_0

    def tresca(self):
        tresca = self.K1**2/(2*np.pi*self.yeild_s**2)*(np.cos(self.theta/2)*(1+np.sin(self.theta/2)))**2
        tresca_0 = self.K1 ** 2 / (2 * np.pi * self.yeild_s ** 2) * (np.cos(0 / 2) * (1 + np.sin(0 / 2))) ** 2
        self.tera = tresca
        return tresca,self.theta ,tresca_0

    def graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.axis('on')
        ax.plot(self.theta, self.von_mises()[0], color="tab:blue", lw=3, label="tresca")
        ax.plot(self.theta, self.tresca()[0], color="tab:red", ls="--", lw=3, label="von mises")
        ax.tick_params(grid_color="white")
        plt.setp(ax.yaxis.get_ticklabels(), visible=False)
        plt.show()

def strain_z_calculation(poisson_ratio, strain_x, strain_y):
    result = -1 * poisson_ratio *(strain_x+strain_y)/(1-poisson_ratio)
    return result

def effective_strain_z(poisson_ratio, strain_x, strain_y, strain_xy):
    strain_z = strain_z_calculation(poisson_ratio,strain_x, strain_y)
    result = 2 * np.sqrt(strain_x**2 + strain_y**2, strain_z**2 - strain_x*strain_y - strain_y*strain_z - strain_y*strain_z + 3*strain_xy**2) / 3
    return result

def effective_strain(strain1, strain2, strain12):
    strain3 = -(strain1 + strain2)
    strain23, strain31 = 0,0
    result = 2 / (3 * 2**0.5) * ((strain1 - strain2)**2 + (strain2 - strain3)**2 +(strain3-strain1)**2 + 6*strain23**2 + 6*strain31**2 + 6*strain12**2)**0.5
    return result

def cart2pol(x, y):
    """out put : dgree, rho"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    degree = np.rad2deg(phi).reshape([-1,1])
    out = np. hstack((rho,degree))
    return out

def pol2cart(rho, degree):
    """in put : degree, rho"""
    phi = np.deg2rad(degree).reshape([-1,1])
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    out = np.hstack((x, y))
    return out

print('Run complete')