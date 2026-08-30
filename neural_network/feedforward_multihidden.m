function [o_hidden, net_hidden, o_output] = feedforward_multihidden(u, w_hidden, w_out)
    num_layers = numel(w_hidden);
    o_hidden   = cell(num_layers, 1);
    net_hidden = cell(num_layers, 1);

    current = u;

    %% hidden layers  (bipolar sigmoid activation)
    for i = 1:num_layers
        n_b          = size(current, 2);
        % prepend bias row of ones
        u_bias       = [ones(1, n_b); current];
        net_hidden{i} = w_hidden{i}' * u_bias;
        o_hidden{i}   = bipolar_sigmoid(net_hidden{i});
        current       = o_hidden{i};
    end

    %% output layer (softmax — for multi-class classification)
    n_b      = size(current, 2);
    u_bias   = [ones(1, n_b); current];
    net_output = w_out' * u_bias;

    % numerically stable softmax: subtract row-max before exp
    net_shifted = net_output - max(net_output, [], 1);
    e_x         = exp(net_shifted);
    o_output    = e_x ./ sum(e_x, 1);
end
