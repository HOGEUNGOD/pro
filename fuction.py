#ASTM- E399 Calculate

def ASTM_E399(Pq,a,B,W):
    """B is thickness, W is width, Pq is stress, A is crack length"""

    f_in = a/W
    f=((2+f_in)*(0.886+4.64*f_in-13.32*(f_in**2)+14.72*(f_in**3)-5.6*(f_in**4)))/((1-f_in)**1.5)
    kq=(Pq*f)/(B*(W**0.5)*1000000)
    return kq

