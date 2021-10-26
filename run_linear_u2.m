clear;
close all;
clc;

% dataset name
dataset_name = 'emotions';
path_main = pwd;
path_save = strcat(path_main, filesep, 'Results');
path_data = strcat(path_main, filesep, 'Datasets');
file_save = strcat(path_save, filesep, 'Results_linear_u2.csv');

% setting
rand('seed', 2^40);
result = [];
K_fold = 3; % cross-validation: 3-fold
lambda = 0.01;
surrogate_loss_option = 'surrogate_loss_u2';

% Loading the dataset
file_name = strcat(path_data, filesep, dataset_name, '.mat');
S = load(file_name);

X_all = S.data;
Y_all = S.target;
Y_all(Y_all < 1) = -1;
% append one feature with all equal to 1 to correspond to the bias
num_feature_origin = size(X_all, 2);
X_all(:, num_feature_origin + 1) = 1;

%normalization
[X_all, PS] = mapstd(X_all', 0, 1);
X_all = X_all';

% Shuffle the dataset
[num_samples, num_feature] = size(X_all);
shuffle_index = randperm(num_samples);
X_all = X_all(shuffle_index, :);
Y_all = Y_all(shuffle_index, :);

% Do cross-validation
hl = zeros(1, K_fold);
sa = zeros(1, K_fold);
ranking_loss = zeros(1, K_fold);

tic;
for index_cv = 1: K_fold
    [X_train, Y_train, X_vali, Y_vali] = CrossValidation(X_all, Y_all, K_fold, index_cv);      
    % train the train dataset and predict the test dataset 
    alpha = 0.01;
    [ W, obj ] = train_logistic_cost_sensitive_SVRG_BB( X_train, Y_train, lambda, alpha, surrogate_loss_option );
    [ pre_F_vali ] = Predict_score( X_vali, W );

    [ Ranking_Loss ] = Evaluation_Metrics( pre_F_vali, Y_vali );

    ranking_loss(index_cv) = Ranking_Loss;

end
toc;
time = double(toc);

RANKING_LOSS_cv_mean = mean(ranking_loss);
RANKING_LOSS_cv_std = std(ranking_loss);
result = [result; RANKING_LOSS_cv_mean RANKING_LOSS_cv_std lambda time];
csvwrite(file_save, result);
