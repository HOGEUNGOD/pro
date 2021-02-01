import cv2
import numpy as np
import pandas as pd


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])
        print("마우스 이벤트발생, x:", x, "y:", y, "클릭 :",len(points), "Contour N.o.:", (len(points)-1)//9+1)


img = cv2.imread('../data/distance_angle/angle56.jpg', 0)

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

points = []
x=[]
y=[]
number = []
num=0

while(True):

    cv2.imshow('image', img)

    k=cv2.waitKey(1)&0xFF
    if k == 27:
        for i in points:
            num += 1
            number = np.append(number, num)
            x = np.append(x, i[0])
            y = np.append(y, i[1])


        break


data_union = {'Contour N.o.': number,'y' :y ,'x': x}
df = pd.DataFrame(data_union)
# df.to_excel('data/distance_angle/60크랙팁.xlsx')
cv2.destroyAllWindows()
