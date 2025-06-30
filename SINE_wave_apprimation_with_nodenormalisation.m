clc; clear all; close all;  

% Number of samples
num_samples = 10000; 

%parameters
amp =2;%amplitude of the wave
freq = 4 ;%frequency of the wave
phshift = 0 ;%the phase difference
numcyc=3;%number of cycles of the wave 

% Generate x values in a specified range
X_raw = linspace(-numcyc * (2 * pi / freq), numcyc * (2 * pi / freq), num_samples)'; % Covers multiple sine cycles  

% Target output (a sine wave)
Y_raw = amp*sin(freq*X_raw + phshift);  

% Normalize X and Y
[X_norm, X_ps] = mapminmax(X_raw'); % Normalize X to [-1,1]
X_norm = X_norm'; % Convert back to column vector

% Shuffle dataset( for better capturing )
data = [X_norm, Y_raw]; 
shuffled_data = data(randperm(size(X_norm,1)), :);  

% Split into training and testing sets
split = round(0.7 * size(shuffled_data,1)); 
train_data = shuffled_data(1:split,:);
test_data = shuffled_data(split+1:end,:);

% Extract input (X) and target (Y)
train_X = train_data(:,1); 
train_Y = train_data(:,2); 
test_X = test_data(:,1); 
test_Y = test_data(:,2); 

% ADDING NOISE TO TRAINING OUTPUT
noise_level = 0.05; % can be adjusted
train_Y_noisy = train_Y + noise_level * randn(size(train_Y));% Adding Gaussian noise  

[Y_norm, Y_ps] = mapminmax(Y_raw'); % Normalize Y to [-1,1]
Y_norm = Y_norm';  

% Convert input X to complex domain (RVFL transformation) and also adding
% different kinds of input feature for better capturing the dataset.
train_data1 = [exp(1i * pi * train_X),exp(2i *pi* train_X),exp(3i *pi* train_X),  train_X.^2, train_X.^3];
test_X1 = [exp(1i * pi * test_X),exp(2i *pi* test_X),exp(3i *pi* test_X), test_X.^2, test_X.^3];
%train_data1 = [exp(1i * pi * train_X)];
%test_X1 = [exp(1i * pi * test_X)];

% RVFL MODEL PARAMETERS
C = 2^(-5); % Regularization parameter
N = size(train_data1, 1); 
num = size(test_X1, 1); 
dim = size(train_data1, 2); 
ActivationFunction = 'atanh'; % Activation function
Times = 60; % Number of times to run per hidden node count  

hidden_nodes_list = 10:10:200;  

% Store results
Traintime = zeros(Times, 1); 
Testingtime = zeros(Times, 1); 
rmstraining = zeros(Times, 1); 
rmstesting = zeros(Times, 1);  

results = []; 
predictions_avg = zeros(size(test_X1,1),1); % Averaged predictions  

% RVFL Training and Testing Loop
for hiddenn = hidden_nodes_list
    predictions_total = zeros(size(test_X1,1), Times); % Store multiple runs for averaging

    for rnd = 1:Times
        % Training
        tic;
        
        weight = 0.1* (2 * rand(hiddenn, dim) - ones(hiddenn, dim));  
        bias = 0.1* rand(1, hiddenn)-1;
        BiasMatrix = repmat(bias, N, 1);  

        tempH = train_data1 * weight.' + BiasMatrix; 
        H1 = atanh(tempH); 
        H = [train_data1, H1];  

        M = pinv(H' * H + (eye(size(H' * H)) * C)); 
        W = M * H' * train_Y_noisy; 
        Train_time = toc;  

        % Calculate training error
        Yout = real(H * W); 
        E2 = train_Y - Yout; 
        rms_training = sqrt(E2.' * E2 / size(train_data1, 1));  

        % Testing
        tic;
        BiasMatrixT = repmat(bias, num, 1); 
        tempH_test = test_X1 * weight.' + BiasMatrixT; 
        H_test1 = atanh(tempH_test); 
        H_test = [test_X1, H_test1];  

        Y_pred = H_test * W; % Predicted normalized Y
        Test_time = toc;  

        E3 = test_Y - Y_pred;  
        rms_testing = sqrt(E3.' * E3 / size(test_X1, 1));  

        % Store results
        Traintime(rnd, 1) = Train_time;  
        Testingtime(rnd, 1) = Test_time;  
        rmstraining(rnd, 1) = rms_training;  
        rmstesting(rnd, 1) = rms_testing;  

        % Store predictions for averaging
        predictions_total(:, rnd) = Y_pred;  
    end  

    % Average predictions over multiple runs
    predictions_avg = mean(predictions_total, 2);  

    % Average results
    Averagermstesting = mean(rmstesting);  
    Averagermstraining = mean(rmstraining);  
    AverageTrainingTime = mean(Traintime);  
    AverageTestingTime = mean(Testingtime);  

    % Store results for plotting
    results = [results; hiddenn, Averagermstesting, Averagermstraining, AverageTrainingTime, AverageTestingTime];  

    % Plot comparison after averaging
    figure(1);  
    clf;  
    plot(X_raw, Y_raw, 'b', 'LineWidth', 2); % Actual sine wave (use raw, not normalized)
    hold on;  
    scatter(mapminmax('reverse', test_X', X_ps), predictions_avg, 10, 'r', 'filled'); 
    grid on;  
    xlabel('x values');  
    ylabel('sin(x)');  
    title(['Comparison of Actual and Predicted Sine Wave (Hidden Nodes = ', num2str(hiddenn), ')']);  
    legend('Actual sin(x)', 'Predicted sin(x)');  
end  

% Plotting Results
figure(2);
% Testing Error Plot
subplot(2, 2, 1);  
plot(results(:, 1), results(:, 2), 'r-o');  
grid on;  
legend('Testing Error');  
xlabel('Number of Hidden Nodes');  
ylabel('Testing Error');  

% Training Error Plot
subplot(2, 2, 2);  
plot(results(:, 1), results(:, 3), 'b-o');  
grid on;  
legend('Training Error');  
xlabel('Number of Hidden Nodes');  
ylabel('Training Error');  

% Average Training Time Plot
subplot(2, 2, 3);  
plot(results(:, 1), results(:, 4), 'g-o');  
grid on;  
legend('Average Training Time');  
xlabel('Number of Hidden Nodes');  
ylabel('Training Time (seconds)');  

% Average Testing Time Plot
subplot(2, 2, 4);  
plot(results(:, 1), results(:, 5), 'm-o');  
grid on;  
legend('Average Testing Time');  
xlabel('Number of Hidden Nodes');  
ylabel('Testing Time (seconds)');  

sgtitle('RVFL Network Performance for Sine Wave Approximation');  

