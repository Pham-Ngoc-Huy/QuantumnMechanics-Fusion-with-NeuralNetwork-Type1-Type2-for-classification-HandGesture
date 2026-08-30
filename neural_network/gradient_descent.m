function [w, b, loss_history] = gradient_descent(x, y, w, b, learning_rate, tolerant)
    N = length(x);
    loss_history = [];
    error = inf;

    while error > tolerant

        y_hat = w .* x + b;

        % compute loss
        loss = sum((y - y_hat).^2) / N;
        loss_history(end+1) = loss; % record loss

        % compute gradients
        % Origin Loss Function => L = 1/N sum(y-(wx+b))**2 | where y_hat = wx + b

        % Apply Chain Rule for the derivative
        dw = (-2/N) * sum(x .* (y - y_hat)); % derivative Loss according to Weight
        db = (-2/N) * sum(y - y_hat); % derivative Loss according to Beta

        w = w - learning_rate * dw;
        b = b - learning_rate * db;

        error = abs(loss);
    end
end
