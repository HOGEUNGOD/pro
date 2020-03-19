from PIL import Image
import os

print(os.getcwd())


for i in range(300,301): ##레인지 범위바꾸기##########################################
    config=random.choice(l5) ##contour 범위 바꿔주기
    print(config)
#이미지 불러오기
    im_real = Image.open('fixreal/5.jpg')
    im_contour = Image.open('fixcontour/FIG%d.png' %config)
#이미지 픽셀 수 추출
    x_real=im_real.size[0]
    y_real=im_real.size[1]
    print(x_real,y_real)
    x_contour=im_contour.size[0]
    y_contour=im_contour.size[1]
    print(x_contour,y_contour)

    x_di=(m5_x-m4_x)/1; y_di=(m5_y-m4_y)/1  #등분바꾸기############################################
    x = int(m5_x - pig_x + (x_di * (i-300)))
    y = int(m5_y - pig_y + (y_di * (i-300)))
    print(i-250)

    im_real.paste(im_contour, (x,y), im_contour)
    im_real.save('%d.png'%i)
