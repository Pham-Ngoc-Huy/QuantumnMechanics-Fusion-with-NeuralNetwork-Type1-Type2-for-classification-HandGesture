% multilayer perceptron (MLP) Neural Networks

% example with 2 layers
% net1 = w_T_1 * x
% o1 = F1(net1) -> F1: stands for activation function
% net2 = w_T_2 * o1
% o2 = F2(net2) -> F2: stands for activation function

%% Activation Function
function [o1,o2,net1,net2] = feedforward_nn(type,u,b1,b2,w1,w2)
    % f1 = @(z) 1 ./ (1+exp(-z)); % @(z) means create fast function
    % f2 = @(z) (1-exp(-z)) ./ (1+exp(-z)); % using for hidden layer
    % hidden layer
    % RELU = @(x) max(0,x);
    u1=[b1;u];
    net1 = w1' * u1;
    o1 = bipolar_sigmoid(net1);
    % output layer
    u2=[b2;o1];
    net2=w2' * u2;
    if type == "regression"
        o2 = bipolar_sigmoid(net2);
    elseif type == "classification"
        o2 = monopolar_sigmoid(net2);
    end
end

%% input 
% x = [0.2;0.2;0.8];
% 
% n_input = 3;
% n_hidden = 4;
% n_output = 1;
% 
% W1 = randn(n_input, n_hidden);
% W2 = randn(n_hidden, n_output);

%% layer creation
% net1 = W1' * x;
% o1 = bipolar_sigmoid(net1);
% 
% net2 = W2' * o1;
% o2 = mono_sigmoid(net2);
% 
% disp('net1 = ');
% disp(net1);
% 
% disp('o1 = ');
% disp(o1);
% 
% disp('net2 = ');
% disp(net2);
% 
% disp('Final output o2 = ');
% disp(o2);