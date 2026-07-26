function [w1,w2] = GD_MLP(type, e, eta, u, o1, net1, net2, w1, w2) % gradient descent for multilayer perception
    % bipolar sigmoid activation function
    df1 = @(x) 2*exp(-x) ./ (1 + exp(-x)).^2; % this is for hidden layer -> dervivative in bipolar sigmoid
    % Hidden layer uses bipolar sigmoid to provide symmetric activation 
    % which improves gradient flow and convergence.
    % Output layer also uses bipolar sigmoid because the target values 
    % are bounded roughly in [-1,1]. For general regression problems, 
    % a linear activation is usually preferred.
    if type == "regression"
        df2 = @(x) 2*exp(-x) ./ (1 + exp(-x)).^2; 
    elseif  type == "classification"
        df2 = @(x) exp(-x) ./ (1 + exp(-x)).^2;
    end

    delta2 = e.*df2(net2);
    delta1 = (w2(2:end,:) * delta2) .* df1(net1);

    % backprogation occur
    rate2 = o1*delta2';
    rate1 = u*delta1';
    w1 = w1 + eta*rate1;
    w2 = w2 + eta*rate2;
end