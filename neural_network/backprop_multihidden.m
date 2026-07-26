function [grad_w_hidden, grad_w_out] = backprop_multihidden(u, T, o_hidden, o_output, w_hidden, w_out)
    num_layers = numel(w_hidden);
    n_b        = size(u, 2);
    % Bipolar sigmoid derivative: f'(net) = 0.5*(1 - o^2)
    % where o = tanh-like bipolar sigmoid output
    % Equivalent form using net: 2*exp(-net)/(1+exp(-net))^2
    % Using output form is faster: df = 0.5*(1 - o.^2)
    df_bipolar = @(o) 0.5 * (1 - o.^2); 

    %% output layer delta  (softmax + CE simplification) 
    delta_out = (o_output - T);   % (n_output x batch) — already the gradient

    %% output layer gradient
    prev_out   = o_hidden{end};
    u_bias_out = [ones(1, n_b); prev_out];            % (n_hidden_last+1 x batch)
    grad_w_out = (u_bias_out * delta_out') / n_b;     % (n_hidden_last+1 x n_output)

    %% backpropagate through hidden layers
    grad_w_hidden = cell(num_layers, 1);

    delta_next = delta_out;    % starts at output delta
    w_next     = w_out;        % weights connecting to the layer above

    for j = num_layers:-1:1
        % Propagate delta back through w_next, skipping bias row (row 1)
        % w_next(2:end,:) has shape (n_out_j x n_out_next)
        % delta_next has shape (n_out_next x batch)
        % result: (n_out_j x batch)
        propagated = w_next(2:end, :) * delta_next;   % strip bias row

        % Element-wise multiply by derivative of bipolar sigmoid at this layer
        % Using post-activation output for efficiency
        delta_j = propagated .* df_bipolar(o_hidden{j});  % (n_out_j x batch)

        % Gradient for this layer's weights
        if j == 1
            prev_input = u;              % first layer receives raw input
        else
            prev_input = o_hidden{j-1}; % deeper layers receive previous hidden output
        end
        u_bias_j      = [ones(1, n_b); prev_input];       % (n_in_j+1 x batch)
        grad_w_hidden{j} = (u_bias_j * delta_j') / n_b;  % (n_in_j+1 x n_out_j)

        % Pass delta and weights upward (toward input) for next iteration
        delta_next = delta_j;
        w_next     = w_hidden{j};
    end
end