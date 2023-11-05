clear all 
clc
clc;
clear all;
close all;
a = VideoReader('v1.mpg');
frames = read(a,1);
I = frames(:,:,:,1);
[r c b] = size(I);
N = 1/(3*r*c);
R=I(:,:,1); Ir_hist= imhist(R); Ir_std= std2(R);
G=I(:,:,2); Ig_hist= imhist(G); Ig_std= std2(G);
B=I(:,:,3); Ib_hist= imhist(B); Ib_std= std2(B);
for i=2:a.NumberOfFrames
disp(['Reading frame: ',num2str(i)]);
J = read(a,i);
R1=J(:,:,1); Jr_hist=imhist(R1); Jr_std= std2(R1);
G1=J(:,:,2); Jg_hist=imhist(G1); Jg_std= std2(G1);
B1=J(:,:,3); Jb_hist=imhist(B1); Jb_std= std2(B1);
Hd(i-1) = 1-N*(sum(min(Ir_hist,Jr_hist))+sum(min(Ig_hist,Jg_hist))+sum(min(Ib_hist,Jb_hist)));
St(i-1) = sqrt((Ir_std-Jr_std)^2+(Ig_std-Jg_std)^2+(Ib_std-Jb_std)^2);
P(i-1) = sqrt(sum(sum((R-R1).^2))+sum(sum((G-G1).^2))+sum(sum((B-B1).^2)));
R=R1; Ir_hist= Jr_hist; Ir_std= Jr_std;
G=G1; Ig_hist= Jg_hist; Ig_std= Jg_std;
B=B1; Ib_hist= Jb_hist; Ib_std= Jb_std;
end
St = St/max(St);
P = P/max(P);
input_test = [Hd;P;St];
noP = 30;
 HiddenNodes=15;      
Dim=5*HiddenNodes+1; 
TrainingNO=150;     
load newfeature.mat
x=features_new(1:150,1:4);
 H2=x(1:150,1);
 H3=x(1:150,2);
 H4=x(1:150,3);
 T=x(1:150,4);
 H2 = H2';
 [xf, PS] = mapminmax(H2);
 I2(:, 1) = xf;
 H3 = H3';
 [xf, PS2] = mapminmax(H3);
 I2(:, 2) = xf;
 H4 = H4';
 [xf, PS3] = mapminmax(H4);
 I2(:, 3) = xf;
 Thelp = T;
 T = T';
 [yf, PS5] = mapminmax(T);
 T = yf;
 T = T';
SearchAgents_no=200; 
Function_name='F1'; 
Max_iteration=500; 
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);
[Best_score,Best_pos,cg_curve]=ALO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
Weights = zeros(1, 4 * HiddenNodes);
Biases = zeros(1, Dim - 4 * HiddenNodes);
for i = 1:noP
        if i <= 4 * HiddenNodes
            Weights(i) = Best_pos(i);
        else
            Biases(i - 4 * HiddenNodes) = Best_pos(i);
        end
        fitness = 0;
        for pp = 1:TrainingNO
            actualvalue = My_FNN(3, HiddenNodes, 1, Weights, Biases, I2(pp, 1), I2(pp, 2), I2(pp, 3));
            if T(pp) == -1
                fitness = fitness + (1 - actualvalue(1))^2;
            end
            if T(pp) == 0
                fitness = fitness + (0 - actualvalue(1))^2;
            end
    if T(pp) == 1
        fitness = fitness + (0 - actualvalue(1))^2;
    end
end

fitness = fitness / TrainingNO;
CurrentFitness = fitness;
 
end
x=input_test;x=x';
n = length(x);
a2 = []
for i=0:floor(n/150)-1
    if ((150*i)+150)<n
        H2=x((150*i)+1:(150*i)+150,1);
        H3=x((150*i)+1:(150*i)+150,2);
        H4=x((150*i)+1:(150*i)+150,3);
        [actualvalue,a1] = test_NN(H2,H3,H4,HiddenNodes,Best_pos,Dim,i);
        a2 = [a2,a1];
        a3=a2;
    else
        H2=x((150*i)+1:end,1);
        H3=x((150*i)+1:end,2);
        H4=x((150*i)+1:end,3);
        [actualvalue,a1] = test_NN(H2,H3,H4,HiddenNodes,Best_pos,Dim,i);
        a2 = [a2,a1];    
    end
end
b=round(a2);        
output = b(:);
plot(output);
tranf = find(output==0);

        


