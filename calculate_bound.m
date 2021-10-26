function [bound_value] = calculate_bound(X_train, Y_train, W, option)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[num_samples, num_labels] = size(Y_train);

bound_value = 0;
W_norm = norm(W, 'fro');
x_norm_bound = 0;
delta = 0.01;

for i = 1: num_samples
    x_norm_bound = max(x_norm_bound, norm(X_train(i,:)));
end

if strcmp(option, 'surrogate_loss_pa')
    [risk, B] = calculate_risk( X_train, Y_train, W, option);
    bound_value = risk + ...
        2*sqrt(2*num_labels)*W_norm*x_norm_bound/sqrt(num_samples) + ...
        3*B*sqrt(log(2/delta)/(2*num_samples));
    
elseif strcmp(option, 'surrogate_loss_u1')
    [risk, B] = calculate_risk( X_train, Y_train, W, option);
    bound_value = risk + ...
        2*sqrt(2)*W_norm*x_norm_bound/sqrt(num_samples) + ...
        3*B*sqrt(log(2/delta)/(2*num_samples));
    bound_value = bound_value * num_labels;
    
elseif strcmp(option, 'surrogate_loss_u2')
    [risk, B] = calculate_risk( X_train, Y_train, W, option);
    bound_value = risk + ...
        2*sqrt(2)*num_labels/(num_labels-1)*W_norm*x_norm_bound/sqrt(num_samples) + ...
        3*B*sqrt(log(2/delta)/(2*num_samples));
    bound_value = bound_value * (num_labels - 1);
    
elseif strcmp(option, 'surrogate_loss_u3')
    [risk, B] = calculate_risk( X_train, Y_train, W, option);
    bound_value = risk + ...
        4*sqrt(num_labels)*W_norm*x_norm_bound/sqrt(num_samples) + ...
        6*B*sqrt(log(2/delta)/(2*num_samples));
    
elseif strcmp(option, 'surrogate_loss_u4')
    [risk, B] = calculate_risk( X_train, Y_train, W, option);
    bound_value = risk + ...
        2*sqrt(2)*num_labels*W_norm*x_norm_bound/sqrt(num_samples) + ...
        3*B*num_labels*sqrt(log(2/delta)/(2*num_samples));
    
end

end

