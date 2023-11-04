function [actualvalue,a1]= test_NN(H2,H3,H4,HiddenNodes,Best_pos,Dim,i)

        H2=H2';
        [xf,PS] = mapminmax(H2);
        I3(:,1)=xf;
 
        H3=H3';
        [xf,PS2] = mapminmax(H3);
        I3(:,2)=xf;

        H4=H4';
        [xf,PS3] = mapminmax(H4);
        I3(:,3)=xf;
 
% %         H5=H5';
% %         [xf,PS4] = mapminmax(H5);
% %         I3(:,4)=xf;

        Rrate=0;
        Weights=Best_pos(1:4*HiddenNodes);
        Biases=Best_pos(4*HiddenNodes+1:Dim);
        for pp=1:150
            actualvalue=My_FNN(3,HiddenNodes,1,Weights,Biases,I3(pp,1),I3(pp,2), I3(pp,3));
            a1(pp) = actualvalue;
        end

end