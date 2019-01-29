%% ==================== Preprocess data in sold_data.csv ====================

fprintf('Prepocessing data - Step 1 - Select data in first 9 columns of raw_input... \n');
% Use csvread to select the first 9 features from the raw datset
% Refer to doc: https://au.mathworks.com/help/matlab/ref/csvwrite.html?s_tid=doc_ta
% Features: PropertyNumber,Sold_Price,Suburb,lat,lon,Bedrooms,Bathrooms,Carparks,AreaSqM
% Note that Suburb is of type string and csvread replaces strings with 0. This is a conscious choice as suburb is NOT being used for modelling at this stage
% There are in all 155890 rows in the raw dataset and we are interested in the first 9 columns 
% Note that row and column start at 0 index

%
data = csvread('data/experiment_2/sold_data.csv', [0,0, 155890,8]);
X = data(:, 1); 
m = length(X); % number of examples

% Write the required features into select_features.csv
% Header: PropertyNumber,Sold_Price,Suburb,lat,lon,Bedrooms,Bathrooms,Carparks,AreaSqM
% Sample Record: 102212828,335000,0,-37.741523,144.989827,2,1,1,1162 
csvwrite('data/experiment_2/select_features.csv', data);
fprintf('Number of examples %f \n', m);


fprintf('Program paused. Press enter to continue.\n');
pause;



%% ==================== Preprocess data in select_features.csv ====================

fprintf('Prepocessing data - Step 2 - Discard Suburb, lat, lon. Rearrange Sold_Price as last column... \n');
% Input Features: PropertyNumber,Sold_Price,Suburb,lat,lon,Bedrooms,Bathrooms,Carparks,AreaSqM
% Discard Suburb,lat,lon
% Re-arrange Sold_Price to be last column
% Scale sold_price and area
% Randomize i.e. shuffle rows in the dataset
% Output Features: PropertyNumber, Bedrooms,Bathrooms,Carparks,AreaSqM, Sold_Price

data = csvread('data/experiment_2/select_features.csv');
propertyNumber = data(:,1);
soldPrice = data(:,2) ./ 100000; % Scale soldPrice - divide each value by 100000
propertyConfig = data(:,6:8); % property config is combination of Bedrooms,Bathrooms,Carparks
area = data(:, 9) ./ 1000;  % Scale Area - divide each value by 1000
%Use Horizontal Concatenation to re-arrange the dataset https://octave.org/doc/v4.4.0/Rearranging-Matrices.html
result = horzcat(propertyNumber, propertyConfig, area, soldPrice);

result = result(randperm(end),:); % Randomize dataset i.e. shuffle rows

csvwrite('data/experiment_2/full_data_set_experiment2.csv', result);
fprintf('Number of examples %f \n', m);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ==================== Split data into Training, Testing, Validation ====================

% TBC %

%% =======================================================================================