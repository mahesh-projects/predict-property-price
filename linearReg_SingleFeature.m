%% Predict house price based on single feature: Linear Regression

%  Instructions
%  ------------
% 
%  This file contains code to predict housing price based on single feature. 
%
% x refers to the area in 100s
% y refers to the profit in $10,000s
%
%% Initialization
clear ; close all; clc

%% ======================= Part 1: Plotting =======================
fprintf('Plotting Data ...\n');
data = load('data/experiment_2/multi_training_data.csv');
X = data(:, 2); y = data(:, 6);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 2: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,2)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 2000;
alpha = 0.001;

% compute and display initial cost
computeCost(X, y, theta)

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: '); 
fprintf('%f %f \n', theta(1), theta(2)); 

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % dont overlay any more plots on this figure

% Predict values for house sizes of area 1785 sq.m 
predict1 = [1, 1.785] * theta;
fprintf('For area = 1785 sq.m, we predict a sold price of %f\n',...
    predict1*100000);
% Predict values for house sizes of area 853 sq.m 
predict2 = [1, 0.853] *theta;
fprintf('For area = 853 sq.m, we predict a sold price of %f\n',...
    predict2*100000);
% Predict values for house sizes of area 190 sq.m 
predict3 = [1, 0.19] * theta;
fprintf('For area = 190 sq.m, we predict a sold price of %f\n',...
    predict3*100000);


fprintf('Program paused. Press enter to continue.\n');
pause;

