function [o_hidden, net_hidden, o_output] = feedforward_nn_hidden_01(X, w_hidden, w_out)
    num_layers = numel(w_hidden);
    o_hidden   = cell(num_layers, 1);
    net_hidden = cell(num_layers, 1);
    input = X;
    for i = 1:num_layers
        n_b = size(input, 2);
        input_b = [ones(1,n_b); input];
        net_h  = w_hidden{i}' * input_b;
        o_h = max(0, net_h); % changing into RELU function since it suitable for membership function [0,1] -> canbe mono sigmoid is cool
        net_hidden{i} = net_h;
        o_hidden{i}   = o_h;
        input = o_h;
    end
    n_b = size(input, 2);
    input_b = [ones(1,n_b); input];
    net_out = w_out' * input_b;
    net_out = net_out - max(net_out,[],1);
    exp_out = exp(net_out);
    o_output = exp_out ./ sum(exp_out, 1);
end
