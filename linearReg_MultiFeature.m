%% Predict house price based on single feature: Linear Regression

%  Instructions
%  ------------
% 
%  This file contains code to predict housing price based on single feature. 
%
% x refers to the area in 100s
% y refers to the profit in $10,000s
%

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('data/experiment_1/multi_training_set.csv');
X = data(:, 2:5);
y = data(:, 6);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.3f %.0f %.0f %.0f], y = %.3f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ============================================================
% Trial 1: Find J for alpha = 0.01; num_iters = 400;

fprintf('Running gradient descent ...\n');


% Init Theta and Run Gradient Descent 
theta = zeros(5, 1);
% ============================================================
% Trial 2: Find J for alpha = 0.1; num_iters = 400;
% ============================================================
% Choose some alpha value
alpha_trial_1 = 0.01;
num_iters_trial_1 = 400;
[theta_trial_1, J_history_trial_1] = gradientDescentMulti(X, y, theta, alpha_trial_1, num_iters_trial_1);


% ============================================================
% Trial 2: Find J for alpha = 0.1; num_iters = 400;
% ============================================================
alpha_trial_2 = 0.1;
num_iters_trial_2 = 400;
[theta_trial_2, J_history_trial_2] = gradientDescentMulti(X, y, theta, alpha_trial_2, num_iters_trial_2);

% ============================================================
% Trial 3: Find J for alpha = 0.001; num_iters = 400;
% ============================================================
alpha_trial_3 = 0.001;
num_iters_trial_3 = 400;
[theta_trial_3, J_history_trial_3] = gradientDescentMulti(X, y, theta, alpha_trial_3, num_iters_trial_3);

% Plot the convergence graph
figure;
hold on;
plot(1:numel(J_history_trial_1), J_history_trial_1, '-b', 'LineWidth', 2);
plot(1:numel(J_history_trial_2), J_history_trial_2, '-r', 'LineWidth', 2);
plot(1:numel(J_history_trial_3), J_history_trial_3, '-k', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold off;

% Display gradient descent's result
fprintf('Theta computed from gradient descent - Trial 1: \n');
fprintf(' %f \n', theta_trial_1);
fprintf('\n');

% Display gradient descent's result
fprintf('Theta computed from gradient descent - Trial 2: \n');
fprintf(' %f \n', theta_trial_2);
fprintf('\n');

% Display gradient descent's result
fprintf('Theta computed from gradient descent - Trial 3: \n');
fprintf(' %f \n', theta_trial_3);
fprintf('\n');

% Set theta to the best forming from the above trials
theta = theta_trial_2;

% Estimate the price of a 1650 sq-m, 3 br house, 2 bathroom, 1 car park
% ====================== YOUR CODE HERE ================================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
X_house = [1.785, 3, 2, 1];
X_house_norm = (X_house - mu) ./ sigma;


price = [1, X_house_norm] *theta; % You should change this

% ======================================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price*100000);

fprintf('Program paused. Press enter to continue.\n');

% Estimate the price of a 853 sq-m, 4 br house, 3 bathroom, 2 car park
% ====================== YOUR CODE HERE ================================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
X_house = [0.853, 4, 3, 2];
X_house_norm = (X_house - mu) ./ sigma;


price = [1, X_house_norm] *theta; % You should change this

% ======================================================================

fprintf(['Predicted price of a 853 sq-m, 4 br house ' ...
         '(using gradient descent):\n $%f\n'], price*100000);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Estimate the price of a 190 sq-m, 3 br house, 2 bathroom, 1 car park
% ====================== YOUR CODE HERE ================================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
X_house = [0.19, 3, 2, 1];
X_house_norm = (X_house - mu) ./ sigma;


price = [1, X_house_norm] *theta; % You should change this

% ======================================================================

fprintf(['Predicted price of a 190 sq-m, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price*100000);

fprintf('Program paused. Press enter to continue.\n');
pause;

