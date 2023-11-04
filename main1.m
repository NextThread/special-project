
clc;
clear all;
close all;
% Reading The Frames %
a = VideoReader('D6.mpg');
frames = read(a,1);
I = frames(:,:,:,1);
[r c b] = size(I);
N = 1/(3*r*c);
% rgb histogram and color std
R=I(:,:,1); Ir_hist= imhist(R); Ir_std= std2(R);
G=I(:,:,2); Ig_hist= imhist(G); Ig_std= std2(G);
B=I(:,:,3); Ib_hist= imhist(B); Ib_std= std2(B);
% % I_gray =rgb2gray(I);
% % e = edge(I_gray,'canny');
% % C1 = nnz(e);  % number of edge pixel
% 3D eucledian distance
for i=2:a.NumberOfFrames
disp(['Reading frame: ',num2str(i)]);
J = read(a,i);
% rgb histogram and coloe std
R1=J(:,:,1); Jr_hist=imhist(R1); Jr_std= std2(R1);
G1=J(:,:,2); Jg_hist=imhist(G1); Jg_std= std2(G1);
B1=J(:,:,3); Jb_hist=imhist(B1); Jb_std= std2(B1);
% %     J_gray = rgb2gray(J);
% %     e = edge(J_gray,'canny');
% %     C2 = nnz(e);  % number of edge pixel
%     color histogram difference
Hd(i-1) = 1-N*(sum(min(Ir_hist,Jr_hist))+sum(min(Ig_hist,Jg_hist))+sum(min(Ib_hist,Jb_hist)));
%     color 3Deucledian of standard deviation
St(i-1) = sqrt((Ir_std-Jr_std)^2+(Ig_std-Jg_std)^2+(Ib_std-Jb_std)^2);
%     3D eucledian of pixel
P(i-1) = sqrt(sum(sum((R-R1).^2))+sum(sum((G-G1).^2))+sum(sum((B-B1).^2)));
%     edge pixel count difference
% %     Ec(i-1) = sqrt((C1-C2)^2);
%     exchanging value for the next run
R=R1; Ir_hist= Jr_hist; Ir_std= Jr_std;
G=G1; Ig_hist= Jg_hist; Ig_std= Jg_std;
B=B1; Ib_hist= Jb_hist; Ib_std= Jb_std;
% %     C1=C2;
end
% normalizing the values
St = St/max(St);
P = P/max(P);
% % Ec = Ec/max(Ec);
input_test = [Hd;P;St];
SearchAgents_no=200; % Number of search agents
Max_iteration=250; % Maximum numbef of iterations

% Load details of the selected benchmark function
fobj=@MLP_Iris
lb=-10;
ub=10;
dim=76;

% Srat traning using GWO
[Best_MSE,Best_NN,cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

% Draw the convergence curve
figure('Position',[500 500 660 290])
semilogy(cg_curve,'Color','r')
hold on
title('Convergence curve')
xlabel('Iteration');
ylabel('MSE');

axis tight
grid off
box on
legend('GWO')

% Calculate the classification rate

% load the dataset and normalization
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

% Load the obtained optimal values for weights and biases
W=Best_NN(1:60);
B=Best_NN(61:76);

for pp=1:150
    actualvalue=My_FNN(3,15,1,W,B,I2(pp,1),I2(pp,2), I2(pp,3));
    if(T(pp)==-1)
        if (actualvalue(1)>=0.95 && actualvalue(2)<0.05 && actualvalue(3)<0.05)
%             Rrate=Rrate+1;
        end
    end
    if(T(pp)==0)
        if (actualvalue(1)<0.05 && actualvalue(2)>=0.95 && actualvalue(3)<0.05)
%             Rrate=Rrate+1;
        end
    end
    if(T(pp)==1)
        if (actualvalue(1)<0.05 && actualvalue(2)<0.05 && actualvalue(3)>=0.95)
%             Rrate=Rrate+1;
        end
    end
end
 x=input_test;x=x';
n = length(x);
a2 = []

for i=0:floor(n/150)-1
    if ((150*i)+150)<n
        H2=x((150*i)+1:(150*i)+150,1);
        H3=x((150*i)+1:(150*i)+150,2);
        H4=x((150*i)+1:(150*i)+150,3);
% %         H5=x((150*i)+1:(150*i)+150,4);
 
        [actualvalue,a1] = test_NN(H2,H3,H4,HiddenNodes,Best_NN,Dim,i);
        a2 = [a2,a1];
        a3=a2;
    else
        H2=x((150*i)+1:end,1);
        H3=x((150*i)+1:end,2);
        H4=x((150*i)+1:end,3);
% %         H5=x((150*i)+1:end,4);
 
        [actualvalue,a1] = test_NN(H2,H3,H4,HiddenNodes,Best_NN,Dim,i);
        a2 = [a2,a1];
       
    end
end
b=round(a2);        
output = b(:);
plot(output);

tranf = find(output==0);
th = 0.15;
for x =2:length(tranf)-1
    if St(tranf(x))>th &&St(tranf(x)-1)<th&&St(tranf(x)+1)<th&&St(tranf(x)+2)<th&&St(tranf(x)-2)<th
        figure();
        subplot(1,2,1);imshow(read(a,tranf(x)));xlabel(tranf(x));
        subplot(1,2,2);imshow(read(a,tranf(x)+1));xlabel(tranf(x)+1);
    end
end



%  Classification_rate=(Rrate/150)*100