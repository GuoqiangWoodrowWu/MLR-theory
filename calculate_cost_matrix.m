function [C] = calculate_cost_matrix(Y, option)
% Calculate the cost matrix for different measures.
%   Detailed explanation goes here
[num_instance, num_label] = size(Y);
C = ones(num_instance, num_label);
Y(Y < 1) = 0;
if strcmp(option, 'surrogate_loss_u3')
    for i = 1: num_instance
        tmp_positive = sum(Y(i,:));
        tmp_negative = num_label - tmp_positive;
        if tmp_positive == 0 || tmp_negative == 0
            C(i,:) = C(i,:) .* 0; 
        else
            for j = 1: num_label
                if Y(i,j) == 1
                    C(i, j) = 1 / tmp_positive;
                else
                    C(i, j) = 1 / tmp_negative;
                end
            end
        end
    end
    
elseif strcmp(option, 'surrogate_loss_u2')
    for i = 1: num_instance
        tmp_positive = sum(Y(i,:));
        tmp_negative = num_label - tmp_positive;
        if tmp_positive == 0 || tmp_negative == 0
            C(i,:) = C(i,:) .* 0; 
        else
            C(i,:) = C(i,:) ./ (tmp_positive * tmp_negative);
        end
    end
elseif strcmp(option, 'surrogate_loss_u4')
    for i = 1: num_instance
        tmp_positive = sum(Y(i,:));
        tmp_negative = num_label - tmp_positive;
        if tmp_positive == 0 || tmp_negative == 0
            C(i,:) = C(i,:) .* 0; 
        else
            C(i,:) = C(i,:) ./ min(tmp_positive, tmp_negative);
        end
    end
    
elseif strcmp(option, 'surrogate_loss_u1')
    %C = ones(num_instance, num_label);
    for i = 1: num_instance
        C(i, :) = C(i, :) ./ num_label;
    end
end
    
end

