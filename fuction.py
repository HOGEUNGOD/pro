import numpy as np
import math

def ASTM_E399(Pq,a,B,W):
    """B is thickness, W is width, Pq is stress, A is crack length"""

    f_in = a/W
    f=((2+f_in)*(0.886+4.64*f_in-13.32*(f_in**2)+14.72*(f_in**3)-5.6*(f_in**4)))/((1-f_in)**1.5)
    kq=(Pq*f)/(B*(W**0.5)*1000000)
    return kq

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
        return k1, f

    def k2(self):
        f = self.a / self.w
        f2 = 7.12 * math.pow(f, 4) - 5.47 * math.pow(f, 3) + 2.79 * math.pow(f, 2) + 2.63 * f + 0.20
        k2 = (math.sqrt(self.a * math.pi) * self.p *f2) / (self.w*self.b)
        return k2, f