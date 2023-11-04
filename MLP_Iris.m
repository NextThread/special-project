
function o=MLP_Iris(solution)

HiddenNodes=15;       %Number of hidden codes
Dim=5*HiddenNodes+1;  %Dimension of masses in GSA
TrainingNO=150;       %Number of training samples
load newfeature.mat
%  x=sortrows(features,2);
x=features_new(1:150,1:4);
 H2=x(1:150,1);
 H3=x(1:150,2);
 H4=x(1:150,3);
% %  H5=x(1:150,4);
 T=x(1:150,4);
 H2=H2';
 [xf,PS] = mapminmax(H2);
 I2(:,1)=xf;
 
 H3=H3';
 [xf,PS2] = mapminmax(H3);
 I2(:,2)=xf;
 
 H4=H4';
 [xf,PS3] = mapminmax(H4);
 I2(:,3)=xf;
 
% %  H5=H5';
% %  [xf,PS4] = mapminmax(H5);
% %  I2(:,4)=xf;
 Thelp=T;
 T=T';
 [yf,PS5]= mapminmax(T);
 T=yf;
 T=T';

for ww=1:60
    W(ww)=solution(1,ww);
end
for bb=61:76
    B(bb-60)=solution(1,bb);
end
fitness=0;
for pp=1:150
    actualvalue=My_FNN(3,15,1,W,B,I2(pp,1),I2(pp,2), I2(pp,3));
    if(T(pp)==-1)
        fitness=fitness+(1-actualvalue(1))^2;
%         fitness=fitness+(0-actualvalue(2))^2;
%         fitness=fitness+(0-actualvalue(3))^2;
    end
    if(T(pp)==0)
        fitness=fitness+(0-actualvalue(1))^2;
%         fitness=fitness+(1-actualvalue(2))^2;
%         fitness=fitness+(0-actualvalue(3))^2;
    end
    if(T(pp)==1)
        fitness=fitness+(0-actualvalue(1))^2;
%         fitness=fitness+(0-actualvalue(2))^2;
%         fitness=fitness+(1-actualvalue(3))^2;
    end
end
fitness=fitness/TrainingNO;
o=fitness;
end