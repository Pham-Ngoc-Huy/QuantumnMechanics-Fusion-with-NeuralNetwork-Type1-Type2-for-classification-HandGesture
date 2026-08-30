function [grad_w_hidden, grad_w_out, delta_defuzz] = fuzzy_backprop(Y_defuzz, T_b, o_hidden, o_output, w_hidden, w_out)

    num_layers = numel(w_hidden);
    n_b        = size(Y_defuzz, 2);

    % Output delta: CE + softmax combined → δ = ŷ − t
    delta = o_output - T_b;                            % (num_class × batch)

    % Output weight gradient
    last_h  = o_hidden{num_layers};
    input_b = [ones(1,n_b); last_h];                  % (hidden_last+1 × batch)
    grad_w_out = input_b * delta' / n_b;              % (hidden_last+1 × num_class)

    grad_w_hidden   = cell(num_layers, 1);
    delta_into_first = [];

    for i = num_layers:-1:1
        % Propagate delta back through current layer's weight matrix
        if i == num_layers
            W_above = w_out(2:end, :);                % remove bias row
        else
            W_above = w_hidden{i+1}(2:end, :);
        end
        % W_above: (n_out_i × n_out_above),  delta: (n_out_above × batch)
        % W_above * delta = (n_out_i × batch)
        delta = W_above * delta;

        % ReLU derivative: zero out where pre-activation ≤ 0
        delta = delta .* (o_hidden{i} > 0);           % (n_out_i × batch)

        % Gradient for w_hidden{i}
        if i == 1
            prev_out = Y_defuzz;                       % (num_class × batch)
        else
            prev_out = o_hidden{i-1};
        end
        input_b = [ones(1,n_b); prev_out];
        grad_w_hidden{i} = input_b * delta' / n_b;

        if i == 1
            delta_into_first = delta;                  % save for fuzzy backprop
        end
    end

    W_h1_no_bias = w_hidden{1}(2:end, :);             % (num_class × hidden1_size)
    delta_defuzz = W_h1_no_bias * delta_into_first;   % (num_class × batch)
end
