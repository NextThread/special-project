% Clearing the workspace and console
clear all
clc
close all

% Create a VideoReader object to read frames from a video file 'D6.mpg'
a = VideoReader('D6.mpg');

% Read the first frame from the video
frames = read(a, 1);
I = frames(:,:,:,1);
[r, c, b] = size(I);
N = 1 / (3 * r * c);

% Initialize variables to store color channel histograms and standard deviations
R = I(:,:,1);
Ir_hist = imhist(R);
Ir_std = std2(R);
G = I(:,:,2);
Ig_hist = imhist(G);
Ig_std = std2(G);
B = I(:,:,3);
Ib_hist = imhist(B);
Ib_std = std2(B);

% Loop through video frames starting from the second frame
for i = 2:a.NumberOfFrames
    disp(['Reading frame: ', num2str(i)]);
    J = read(a, i);
    
    % Extract color channel histograms and standard deviations for the current frame
    R1 = J(:,:,1);
    Jr_hist = imhist(R1);
    Jr_std = std2(R1);
    G1 = J(:,:,2);
    Jg_hist = imhist(G1);
    Jg_std = std2(G1);
    B1 = J(:,:,3);
    Jb_hist = imhist(B1);
    Jb_std = std2(B1);
    
    % Calculate Histogram Difference (Hd), Color Standard Deviation Difference (St), and Pixel-wise Frame Difference (P)
    Hd(i-1) = 1-N*(sum(min(Ir_hist,Jr_hist))+sum(min(Ig_hist,Jg_hist))+sum(min(Ib_hist,Jb_hist)));
    St(i-1) = sqrt((Ir_std-Jr_std)^2+(Ig_std-Jg_std)^2+(Ib_std-Jb_std)^2);
    P(i-1) = sqrt(sum(sum((R-R1).^2))+sum(sum((G-G1).^2))+sum(sum((B-B1).^2)));
    
    % Update the reference frame and color statistics for the next iteration
    R = R1;
    Ir_hist = Jr_hist;
    Ir_std = Jr_std;
    G = G1;
    Ig_hist = Jg_hist;
    Ig_std = Jg_std;
    B = B1;
    Ib_hist = Jb_hist;
    Ib_std = Jb_std;
end

% Normalize St and P by dividing by their maximum values
St = St / max(St);
P = P / max(P);

% Load data from 'newfeature.mat' for neural network training
noP = 30;
HiddenNodes = 15;
input_test = [Hd;P;St];
Dim = 5 * HiddenNodes + 1;
TrainingNO = 150;
load newfeature.mat
x = features_new(1:150, 1:4);

% Extract data and normalize it
H2 = x(1:150, 1)';
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

% Specify parameters for optimization
SearchAgents_no = 200;
Function_name = 'F1';
Max_iteration = 500;

% Get function details for optimization
[lb, ub, dim, fobj] = Get_Functions_details(Function_name);

% Call optimization function (ALO) to find the best weights and biases
[Best_score, Best_pos, cg_curve] = ALO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj);

% Initialize arrays to store weights and biases
Weights = zeros(1, 4 * HiddenNodes);
Biases = zeros(1, Dim - 4 * HiddenNodes);

% Loop to assign weights and biases to the neural network
for i = 1:noP
    if i <= 4 * HiddenNodes
        Weights(i) = Best_pos(i);
    else
        Biases(i - 4 * HiddenNodes) = Best_pos(i);
    end
    
    % Initialize fitness for neural network training
    fitness = 0;
    
    % Loop through training data
    for pp = 1:TrainingNO
        actualvalue = My_FNN(3, HiddenNodes, 1, Weights, Biases, I2(pp, 1), I2(pp, 2), I2(pp, 3));
        
        % Calculate fitness based on expected output
        if T(pp) == -1
            fitness = fitness + (1 - actualvalue(1))^2;
        elseif T(pp) == 0
            fitness = fitness + (0 - actualvalue(1))^2;
        elseif T(pp) == 1
            fitness = fitness + (0 - actualvalue(1))^2;
        end
    end
    
    % Normalize fitness by dividing by the number of training samples
    fitness = fitness / TrainingNO;
    CurrentFitness = fitness;
end

% Prepare the input data for testing the neural network
x = input_test;
x = x';
n = length(x);

% Initialize arrays for storing test results
a2 = [];
a3 = [];

% Loop for testing the neural network with input data
for i = 0:floor(n/150) - 1
    if ((150 * i) + 150) < n
        H2 = x((150 * i) + 1:(150 * i) + 150, 1);
        H3 = x((150 * i) + 1:(150 * i) + 150, 2);
        H4 = x((150 * i) + 1:(150 * i) + 150, 3);
        [actualvalue, a1] = test_NN(H2, H3, H4, HiddenNodes, Best_pos, Dim, i);
        a2 = [a2, a1];
        a3 = a2;
    else
        H2 = x((150 * i) + 1:end, 1);
        H3 = x((150 * i) + 1:end, 2);
        H4 = x((150 * i) + 1:end, 3);
        [actualvalue, a1] = test_NN(H2, H3, H4, HiddenNodes, Best_pos, Dim, i);
        a2 = [a2, a1];
    end
end

% Round the network output to get the final result
b = round(a2);
output = b(:);

% Plot the network's output
plot(output);

% Find indices where the output is equal to 0
% Find indices where the output is equal to 0
tranf = find(output == 0);

% Additional processing or actions after finding indices where output is 0 can be added here

% Display a message indicating the completion of the job
disp('Job done');
