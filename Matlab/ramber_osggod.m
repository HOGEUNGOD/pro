clear all
L_data=xlsread('al7075.xlsx','A1:B19863')
%ML_data=xlsread('stressstrain.xlsx','J2:K2001')
%%******************1st
stre=L_data(:,2)-L_data(1,2);
str=(L_data(:,1)-L_data(1,1));

n=17;
alpha= 0.585
E= 71.7*1000;
So=535.435;

Tstre=stre.*(1+str);
Tstr=log(1+str);



e=Tstre./E+alpha*Tstre./E.*(Tstre/So).^(n-1)
%%%%%%%******************2st
%ML_s=ML_data(:,1)
%ML_In=ML_data(:,2)
%MLE=5e+03;
%alpha1=0.003

%ML=ML_s./MLE+alpha1*ML_s./MLE.*(ML_s/So).^(n-1)

figure (1)
plot(str,stre)
hold on

plot(e,Tstre,'LineWidth',3)
hold on
plot(Tstr,Tstre)
hold off

%figure (2)
%plot(ML_In,ML_s)
%hold on
%plot(ML,ML_s)
%hold off