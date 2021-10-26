function [ predict_F ] = Predict_score( X, W )
% predict the linear model
% Input: size(X) = [n_instances, n_features] 
%        size(W) = [n_features, n_labels]
% Output: size(predict_Label) = [n_instances, n_labels], 
%         size(predict_F) = [n_instances, n_labels], 
%         predict_F \in R
    
    predict_F = X * W;
end