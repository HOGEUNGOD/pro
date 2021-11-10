clear all

clear al

c_im = imread('./data/ff000001.jpg')
c_im_sub = c_im
r_im = imread('./data/ff000001.jpg')
r_im_sub = r_im(297:309, 54:65)


C = normxcorr2(r_im_sub,c_im_sub)
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
