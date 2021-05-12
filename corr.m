clear all

clear al
x = 83
y = 368
width = 19 
hight =19


r_im = imread('ff_00361.jpg')
r_im_sub = r_im(y:y+hight,x:x+width)
c_im = imread('ff_05881.jpg')
c_im_sch = c_im(151:564,69:140)


C = normxcorr2(r_im_sub,c_im_sch)
[ypeak,xpeak]=find(C==max(C(:)))
max_corr = max(C(:))
yoffset = ypeak-size(r_im_sub,1)
xoffset = xpeak-size(r_im_sub,2)



figure(1)
surf(C)
colormap jet
shading faceted


figure (2)
imshow(c_im_sch)
drawrectangle(gca,'Position',[xoffset,yoffset,size(r_im_sub,2),size(r_im_sub,1)], 'FaceAlpha',0)
figure (3)
imshow(r_im)
drawrectangle(gca,'Position',[x,y,size(r_im_sub,2),size(r_im_sub,1)], 'FaceAlpha',0)
