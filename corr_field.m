clear all


reference = imread('./data/ff000001.jpg');
deformed = imread('./data/ff000009.jpg');
reference_roi = reference(368:384, 100:116)
montage({reference_roi,deformed})
%% 

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


