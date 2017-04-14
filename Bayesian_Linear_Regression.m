%% 
clear;
clc;

%% load data 
fprintf('start to load data\n');

load('train.mat');
load('test.mat');

fprintf('load train and test successfully\n');
fprintf('=================================\n');
%% format data
fprintf('start to format data\n');

X_train = double(train_x);
Y_train = double(train_labels');
X_test = double(test_x);
Y_test = double(test_labels');

% size of original data
[N_train, D_train] = size(X_train);
[N_test, ~] = size(X_test);

fprintf('format data successfully\n');
fprintf('========================\n');
%% split data
fprintf('start to sample data\n');
x_train = X_train;
y_train = Y_train;
x_test = X_test;
y_test = Y_test;

y_train( y_train < 0.5 ) = -1;
y_test( y_test<0.5 ) = -1;

% size of data
[n, d] = size(x_train);
[n_test, d_test] = size(x_test);

fprintf('sample data successfully\n');

% clear original data
clear -regexp ^X ^Y ^N ^D;
clear test_labels test_x train_labels train_x;
fprintf('clear original data successfully\n');
fprintf('===================================\n');
%% SVD on training data
fprintf('start to perform SVD\n');

[U,S,V] = svd( x_train );

clear x_train;

n_features = 200;
S(:,n_features+1:end) = 0;
x_train = U*S*V';

clear U V
fprintf('SVD performed successfully\n');
fprintf('==============================\n');
%% parameters
fprintf('start to choose parameters\n');
sigma_noise = 0.1^2;

% prior for weights
w_prior_mu = zeros(d,1);
w_prior_sigma = eye(d);

fprintf('choose parameters successfully\n');
fprintf('==============================\n');
%% posterior
fprintf('start to compute posterior for weights\n');

w_posterior_mu = ...
    w_prior_mu + ...
    w_prior_sigma * x_train' / ...
    (x_train * w_prior_sigma * x_train' + sigma_noise * eye(n)) * ...
    (y_train - x_train * w_prior_mu);

w_posterior_sigma = ...
    w_prior_sigma - ...
    w_prior_sigma * x_train' / ...
    (x_train * w_prior_sigma * x_train' + sigma_noise * eye(n)) * ...
    x_train * w_prior_sigma;

fprintf('ccompute posterior for weights successfully\n');
fprintf('=============================================\n');
%% predictions
fprintf('start to make predictions\n');

y_star_posterior_mu = ...
    x_test * w_posterior_mu;
y_star_posterior_sigma = ...
    x_test * w_posterior_sigma * x_test' + ...
    sigma_noise * eye(n_test);


fprintf('predictions made successfully\n');
fprintf('===============================\n');
%% result
fprintf('start to get result\n');

predictions = y_star_posterior_mu;
predictions( predictions > 0 ) = 1;
predictions( predictions < 0 ) = -1;
accuracy = sum(predictions == y_test)/n_test;

fprintf('result got successfully\n');