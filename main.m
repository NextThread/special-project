
clear all 
clc

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
% input_test = [0.429079861111111;0.168927410484477;0.653405130347165];
 %% /////////////////////////////////////////////FNN initial parameters//////////////////////////////////////
noP = 30;
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
SearchAgents_no=200; % Number of search agents

Function_name='F1'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

Max_iteration=500; % Maximum numbef of iterations

% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);

[Best_score,Best_pos,cg_curve]=ALO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
for i = 1:noP
        for ww=1:(4*HiddenNodes)
            Weights(ww)=Best_pos(i,ww);
        end
        for bb=4*HiddenNodes+1:Dim
            Biases(bb-(4*HiddenNodes))=Best_pos(i,bb);
        end              
        fitness=0;
        for pp=1:TrainingNO
            actualvalue=My_FNN(3,HiddenNodes,1,Weights,Biases,I2(pp,1),I2(pp,2),I2(pp,3));
            if(T(pp)==-1)
                fitness=fitness+(1-actualvalue(1))^2;
%                 fitness=fitness+(0-actualvalue(2))^2;
%                 fitness=fitness+(0-actualvalue(3))^2;
            end
            if(T(pp)==0)
                fitness=fitness+(0-actualvalue(1))^2;
%                 fitness=fitness+(1-actualvalue(2))^2;
%                 fitness=fitness+(0-actualvalue(3))^2;   
            end
            if(T(pp)==1)
                fitness=fitness+(0-actualvalue(1))^2;
%                 fitness=fitness+(0-actualvalue(2))^2;
%                 fitness=fitness+(1-actualvalue(3))^2;              
            end
        end
        fitness=fitness/TrainingNO; %Equation (5.4)
        CurrentFitness(i) = fitness;     
        
%         if(Best_score>fitness)
%             Best_score=fitness;
% %             gBest=Best_pos(i,:);
%         end  
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
 
        [actualvalue,a1] = test_NN(H2,H3,H4,HiddenNodes,Best_pos,Dim,i);
        a2 = [a2,a1];
        a3=a2;
    else
        H2=x((150*i)+1:end,1);
        H3=x((150*i)+1:end,2);
        H4=x((150*i)+1:end,3);
% %         H5=x((150*i)+1:end,4);
 
        [actualvalue,a1] = test_NN(H2,H3,H4,HiddenNodes,Best_pos,Dim,i);
        a2 = [a2,a1];
       
    end
end
b=round(a2);        
output = b(:);
plot(output);

tranf = find(output==0);
% figure('Position',[500 500 660 290])
% %Draw search space
% subplot(1,2,1);
% func_plot(Function_name);
% title('Test function')
% xlabel('x_1');
% ylabel('x_2');
% zlabel([Function_name,'( x_1 , x_2 )'])
% grid off
% 
% %Draw objective space
% subplot(1,2,2);
% semilogy(cg_curve,'Color','r')
% title('Convergence curve')
% xlabel('Iteration');
% ylabel('Best score obtained so far');
% 
% axis tight
% grid off
% box on
% legend('ALO')
% 
% display(['The best solution obtained by ALO is : ', num2str(Best_pos)]);
% display(['The best optimal value of the objective funciton found by ALO is : ', num2str(Best_score)]);

        



