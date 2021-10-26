function [W, obj] = train_logistic_rank_SVRG_BB( X, Y, lambda_1, alpha )
%UNTITLED2 Summary of this function goes here
%   paiwise ranking loss with base loss is logistic
%   alpha: learning_rate
%   lambda_1 for l2 norm
    [num_instance, num_feature] = size(X);
    num_label = size(Y, 2);
    W = zeros(num_feature, num_label);
    
    % Do serveral SGD steps first
    for i = 1: 10
        index = randi(num_instance);
        GD_one = calculate_one_gradient(X(index,:), Y(index,:), W, lambda_1);
        W = W - alpha * GD_one;
    end
    
    num_s = 30;
    %num_s = 0;
    obj = zeros(num_s, 1);
    m = 2 * num_instance;
    epsilon = 10^-6;
    for i = 1: num_s
        W1 = W;
        fG1 = calculate_all_gradient(X, Y, W1, lambda_1);
        if i > 1
            if i > 2 && abs(obj(i-1, 1) - obj(i-2, 1)) / obj(i-2, 1) <= epsilon
                break;
            end
            alpha = norm(W1-W0, 'fro')^2 / trace((W1-W0)'*(fG1-fG0)) / m;
        end
        fG0 = fG1;
        W0 = W1;
        for j = 1: m
            index = randi(num_instance);
            GD_one = calculate_one_gradient(X(index,:), Y(index,:), W, lambda_1);
            GD_ = calculate_one_gradient(X(index,:), Y(index,:), W1, lambda_1);
            W = W - alpha * (GD_one - GD_ + fG1);
            %if isnan(W)
            %    return;
            %end
        end
        obj(i,1) = calculate_objective_function(X, Y, W, lambda_1);
        fprintf('Step %d: the objective function value is %.5f\n', i, obj(i,1));
    end
end

function [f_value] = calculate_objective_function(X, Y, W, lambda_1)
    f_value = 0.5 * lambda_1 * norm(W, 'fro')^2;
    [num_instance, num_class] = size(Y);

    f_value_rank = calculate_fValue_ranking_loss( X, Y, W);

    f_value = f_value + 1 / num_instance * f_value_rank;
end


function [grad] = calculate_all_gradient(X, Y, W, lambda_1)
    [num_instance, num_class] = size(Y);
    num_feature = size(X, 2);

    grad = lambda_1 * W;
    Z_m = zeros(num_feature, num_class);
    
    grad_rank = Z_m;
    
    for i = 1: num_instance
        grad_rank = grad_rank + calculate_one_gradient_ranking_loss(...
            X(i,:), Y(i,:), W);
    end
    
    grad = grad + grad_rank / num_instance;
end

function [grad_one] = calculate_one_gradient(x, y, W, lambda_1)
% input: size(x) = [1, num_feature], size(y) = [1, num_class]
% Calculate logistic loss gradient
%     [num_feature, num_class] = size(W);
%     Z_m = zeros(num_feature, num_class);
    grad_one = lambda_1 * W;

    grad_rank = calculate_one_gradient_ranking_loss(x, y, W);

    grad_one = grad_one + grad_rank;

end

function [ W_gradient ] = calculate_one_gradient_ranking_loss( x, y, W )
    [n_features, n_labels] = size(W);
    W_gradient = zeros(n_features, n_labels);
    
    num_positive = sum(y > 0);
    num_negative = n_labels - num_positive;
    
    if num_positive == 0 || num_negative == 0
        return;
    end
    
    tmp_gradient = zeros(n_features, n_labels);
    for j = 1: n_labels
        if y(j) > 0
            q_list = find(y < 0);
            for q = 1: length(q_list)
                tmp_gradient(:, j) = tmp_gradient(:, j) + (-x') * (1 / (1 + exp(dot(W(:,j)-W(:,q_list(q)), x))));
            end
        else
            p_list = find(y > 0);
            for p = 1: length(p_list)
                tmp_gradient(:, j) = tmp_gradient(:, j) + x' * (1 / (1 + exp(dot(W(:,p_list(p))-W(:,j), x))));
            end
        end
    end
    W_gradient = W_gradient + tmp_gradient / (num_positive * num_negative);
end

function [ f_value ] = calculate_fValue_ranking_loss( X, Y, W )
    [n_samples, ~] = size(Y);
    f_value = 0;
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
%                 log(1 + exp(dot(X(i,:), - W(:,p_list(p)) + W(:,q_list(q)))))
                tmp_value = tmp_value + log(1 + exp(dot(X(i,:), - W(:,p_list(p)) + W(:,q_list(q)))));
            end
        end
        f_value = f_value + tmp_value / (num_positive * num_negative);
    end
end