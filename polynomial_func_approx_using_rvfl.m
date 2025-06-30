clc; clear all; close all;

% Number of samples
num_samples = 10000;



% Generate X values
X_raw = linspace(-1.5, 1.5, num_samples)';
% Polynomial Coefficients
a = 0.12; b = -0.3; c = 0.05; d = 0.2; e = -0.45; f = 0.13; g = 0.7; h = 0.2;
% Target polynomial output
Y_raw = a*X_raw.^7 + b*X_raw.^6 + c*X_raw.^5 + d*X_raw.^4 + e*X_raw.^3 + f*X_raw.^2 + g*X_raw + h;

% Normalize X and Y
[X_norm, X_ps] = mapminmax(X_raw'); X_norm = X_norm';
[Y_norm, Y_ps] = mapminmax(Y_raw'); Y_norm = Y_norm';

% Shuffle dataset
data = [X_norm, Y_raw]; 
shuffled_data = data(randperm(size(X_norm,1)), :);

% Split into train/test
split = round(0.7 * size(shuffled_data,1)); 
train_data = shuffled_data(1:split,:);
test_data = shuffled_data(split+1:end,:);

% Extract input/output
train_X = train_data(:,1); train_Y = train_data(:,2);
test_X = test_data(:,1); test_Y = test_data(:,2);

% Add Gaussian noise to training output
noise_level = 0.05;
train_Y_noisy = train_Y + noise_level * randn(size(train_Y));

% Create abstract features using nonlinear transforms
train_data1 = [exp(1i*pi*train_X)];
test_X1 = [exp(1i*pi*test_X)];
% RVFL PARAMETERS
C = 2^(-6); 
N = size(train_data1, 1);
num = size(test_X1, 1);
dim = size(train_data1, 2);
ActivationFunction = 'atanh';
Times = 60;

hidden_nodes_list = 10:10:100;
Traintime = zeros(Times, 1);
Testingtime = zeros(Times, 1);
rmstraining = zeros(Times, 1);
rmstesting = zeros(Times, 1);

results = [];
predictions_avg = zeros(size(test_X1,1),1);

for hiddenn = hidden_nodes_list
    predictions_total = zeros(size(test_X1,1), Times);

    for rnd = 1:Times
        tic;
        weight = 0.1* (2 * rand(hiddenn, dim) - ones(hiddenn, dim));
        bias = 0.1* rand(1, hiddenn) - 1;
        BiasMatrix = repmat(bias, N, 1);

        tempH = train_data1 * weight.' + BiasMatrix;
        H1 = atanh(tempH);
        H = [train_data1, H1];

        M = pinv(H' * H + (eye(size(H' * H)) * C));
        W = M * H' * train_Y_noisy;
        Train_time = toc;

        Yout = real(H * W);
        E2 = train_Y - Yout;
        rms_training = sqrt(E2.' * E2 / size(train_data1, 1));

        tic;
        BiasMatrixT = repmat(bias, num, 1);
        tempH_test = test_X1 * weight.' + BiasMatrixT;
        H_test1 = atanh(tempH_test);
        H_test = [test_X1, H_test1];

        Y_pred = H_test * W;
        Test_time = toc;

        E3 = test_Y - Y_pred;
        rms_testing = sqrt(E3.' * E3 / size(test_X1, 1));

        Traintime(rnd, 1) = Train_time;
        Testingtime(rnd, 1) = Test_time;
        rmstraining(rnd, 1) = rms_training;
        rmstesting(rnd, 1) = rms_testing;

        predictions_total(:, rnd) = Y_pred;
    end

    predictions_avg = mean(predictions_total, 2);

    Averagermstesting = mean(rmstesting);
    Averagermstraining = mean(rmstraining);
    AverageTrainingTime = mean(Traintime);
    AverageTestingTime = mean(Testingtime);

    results = [results; hiddenn, Averagermstesting, Averagermstraining, AverageTrainingTime, AverageTestingTime];

    % Plot comparison
    figure(1);
    clf;
    plot(X_raw, Y_raw, 'b', 'LineWidth', 2);
    hold on;
    plot(xlim, [0 0], 'k--');  % X-axis
    plot([0 0], ylim, 'k--');  % Y-axis
    scatter(mapminmax('reverse', test_X', X_ps), predictions_avg, 10, 'r', 'filled');
    grid on;
    xlabel('x values');
    ylabel('Polynomial Output');
    title(['Actual vs Predicted Polynomial (Hidden Nodes = ', num2str(hiddenn), ')']);
    legend('Actual Polynomial', 'Predicted');
end

% Plotting Performance Metrics
figure(2);
subplot(2,2,1); plot(results(:,1), results(:,2), 'r-o'); grid on;
xlabel('Hidden Nodes'); ylabel('Testing Error'); title('Testing Error');

subplot(2,2,2); plot(results(:,1), results(:,3), 'b-o'); grid on;
xlabel('Hidden Nodes'); ylabel('Training Error'); title('Training Error');

subplot(2,2,3); plot(results(:,1), results(:,4), 'g-o'); grid on;
xlabel('Hidden Nodes'); ylabel('Training Time'); title('Avg Training Time');

subplot(2,2,4); plot(results(:,1), results(:,5), 'm-o'); grid on;
xlabel('Hidden Nodes'); ylabel('Testing Time'); title('Avg Testing Time');

sgtitle('RVFL Performance on Polynomial Function');
