function [o1,o2,net1,net2] = FeedForward_NN(u,b1,b2,w1,w2)
% FEEDFORWARD_NN Summary of this function goes here
%   Detailed explanation goes here
% Input:
% u: input
% b1, b2: biases
% w1, w2: weight
% Output: o1,o2,net1,net2
f1 = @(x) (1-exp(-x))./(1+exp(-x));
% f2 = @(x) 1./(1+exp(-x));
f2 = @(x) (1-exp(-x))./(1+exp(-x));
% hidden layer
u1 = [b1;u];
net1 = w1'*u1;
o1 = f1(net1);
% outlayer
u2 = [b2;o1];
net2 = w2'*u2;
o2 = f2(net2);
end

% net1 = w1'*x
% o1 = F1(net1)
% net2 = w2'*o1
% o2 = F2(net2)
