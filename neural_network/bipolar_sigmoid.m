function output = bipolar_sigmoid(var)
    % output = (1-exp(-var))./(1+exp(-var));
    output = tanh(var); % support function may directly faster in Matlab
end
