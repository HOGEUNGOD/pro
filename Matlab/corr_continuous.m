clear all


reference = imread('./data/ff00001.jpg');
deformed = imread('./data/ff12002.jpg');
reference_roi = reference(368:386, 103:121)
reference_roi_good = reference(368:386, 50:68)

c = normxcorr2(reference_roi,deformed);
c_good = normxcorr2(reference_roi_good,deformed);

[ypeak,xpeak] = find(c==max(c(:)));
[ypeak_good,xpeak_good] = find(c_good==max(c_good(:)));

disp(ypeak_good)
disp(ypeak)

