import numpy as np
import math

def ASTM_E399(Pq,a,B,W):
    """B is thickness, W is width, Pq is stress, A is crack length"""

    f_in = a/W
    f=((2+f_in)*(0.886+4.64*f_in-13.32*(f_in**2)+14.72*(f_in**3)-5.6*(f_in**4)))/((1-f_in)**1.5)
    kq=(Pq*f)/(B*(W**0.5)*1000000)
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

K = K_deviator(620.1899, 0.000228, 60)
print(K)
