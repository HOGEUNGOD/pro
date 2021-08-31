clear all

reference = imread('./data/num_00001.jpg');
deformed = imread('./data/num_08002.jpg');
%deformed = deformed(210:458, 52:158)
reference_roi = reference(367-2:386-2, 111:129)
montage({reference_roi,deformed})


c = normxcorr2(reference_roi,deformed);
max_corr = max(c(:))
surf(c)
shading flat
%%
[ypeak,xpeak] = find(c==max(c(:)));
%%
yoffSet = ypeak-size(reference_roi,1);
xoffSet = xpeak-size(reference_roi,2);
%%
imshow(deformed)
drawrectangle(gca,'Position',[xoffSet,yoffSet,size(reference_roi,2),size(reference_roi,1)], ...
    'FaceAlpha',0);


