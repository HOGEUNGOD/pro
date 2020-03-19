import os
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""
DATA PATH
ASTM Kp 구하기위한 crack 하고 Pq값이 있는 데이터 파일
A열에 크랙길이, B열에 힘
2행 부터 읽음 
"""
path = '321.xlsx'

"""specimen dat CHANGE"""




#ASTM- E399 Calculate
def ASTM_E399(Pq,a):
    B = 0.003
    Bn = 0.003
    W = 0.032

    f_in = a/W
    f=((2+f_in)*(0.886+4.64*f_in-13.32*(f_in**2)+14.72*(f_in**3)-5.6*(f_in**4)))/((1-f_in)**1.5)
    kq=(Pq*f)/(B*(W**0.5)*1000000)
    return kq


#take data
wb = openpyxl.load_workbook(path)
sheet = wb['Sheet1']
j=0


for i in sheet.rows:
    j=j+1
    print(i)
    a=i[0].value
    print(a)
    Pq=i[1].value
    kp=ASTM_E399(Pq,a)
    sheet.cell(row=j, column=3).value = kp


wb.save('C:/Users/ghrms/PycharmProjects/untitled1/321.xlsx')