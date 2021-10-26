function [risk, B] = calculate_risk( X, Y, W, option)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
[ pre_F ] = Predict_score( X, W );
if strcmp(option, 'surrogate_loss_pa')
    [risk, B] = calculate_surrogate_pairwise_risk( pre_F, Y );
else
    [risk_pa, B_pa] = calculate_surrogate_pairwise_risk( pre_F, Y );
    risk_pa
    [risk, B] = calculate_surrogate_univarite_risk( pre_F, Y, option );
end
end


function [ risk, B ] = calculate_surrogate_pairwise_risk( pre_F, Y )
    [n_samples, ~] = size(Y);
    risk = 0;
    B = 0;
    for i = 1: n_samples
        p_list = find(Y(i,:) > 0);
        q_list = find(Y(i,:) < 0);
        num_positive = length(p_list);
        num_negative = length(q_list);
        
        if num_positive == 0 || num_negative == 0
            continue;
        end
        
        tmp_value = 0;
        for p = 1: num_positive
            for q = 1: num_negative
                tmp_loss = log(1 + exp(- pre_F(i,p_list(p)) + pre_F(i,q_list(q))));
                tmp_value = tmp_value + tmp_loss;
                if tmp_loss > B
                    B = tmp_loss;
                end 
            end
        end
        risk = risk + tmp_value / (num_positive * num_negative);
    end
    risk = risk / n_samples;
end


function [ risk, B ] = calculate_surrogate_univarite_risk( pre_F, Y, option )
    C = calculate_cost_matrix(Y, option);
    
    [num_instance, num_class] = size(Y);
    
    I = ones(num_instance, num_class);
    %Z = zeros(num_instance, num_class);
    tmp = log(I + exp(-Y .* pre_F));
    
    B = max(max(tmp));
    
    temp1 = tmp .* C;
    
    f_value_point = sum(sum(temp1, 2));
    %f_value_point = f_value_point / num_class;
    
    risk = 1 / num_instance * f_value_point;
end
