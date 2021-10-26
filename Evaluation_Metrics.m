function [ Ranking_Loss ] = Evaluation_Metrics( pre_F, Y )
%UNTITLED4 Evaluate the model for many metrics
%   Detailed explanation goes here

    cd('./measures');
    Ranking_Loss = Ranking_loss(pre_F, Y);
    cd('../');
end
