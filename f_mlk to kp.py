#ML로 구한 K1 값을 공식에 대입하여 kp값으로 바꾸어줌

import openpyxl

def mlk2kp(mlk):
    n = 17; So = 535.435; v = 0.33; al = 0.585; In = 2.78; E = 71.7 * 1000
    kp=((mlk**2)/(al*(So**2)*In))**(1/(n+1))

    return kp


path = '321.xlsx'
wb = openpyxl.load_workbook(path)
sheet = wb['Sheet1']
j=0

for i in sheet.rows:
    j=j+1
    mlk=i[2].value
    mlkp = mlk2kp(mlk)
    sheet.cell(row=j, column=4).value = mlkp


wb.save('C:/Users/ghrms/PycharmProjects/untitled1/321.xlsx')
## material


