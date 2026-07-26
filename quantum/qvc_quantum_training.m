%% clean
clc;
clear;

%% raw data -> self given dataset
file_name = 'data_qc.csv';
dataset = readtable(file_name);

%% number of qubit define:
% since we have 2 var : x1,x2 -> qubit use = 2
% number of qubit use will = the number of features 
% label is not feature -> it is the output
qubit = 2;

%% call out the data variable
% data input
x1 = dataset.x1;
x2 = dataset.x2;

% label
label = dataset.label;

%% normalize
max_val = max(max(x1),max(x2));
x1_normalize = x1/max_val * pi;
x2_normalize = x2/max_val * pi;

%% define initial quantum state |00>
state0 = [1;0;0;0];

%% Define rotation matix
RY1 = cell(length(x1_normalize),1); % this is use `cell` for store `matrix` as single element 
RY2 = cell(length(x2_normalize),1);
U = cell(length(x1_normalize),1);
for i = 1:length(x1_normalize)
    RY1{i} = rotation_matrix(x1_normalize(i));
    RY2{i} = rotation_matrix(x2_normalize(i));
    % Encoding circuit
    U{i} = kron(RY1{i}, RY2{i});
end

%% parameter
theta1 = 0.5;
theta2 = -0.3;
eta = 0.1; % delta increse = 0.1
%% Observable
Z = [1,0;0,-1];
I = eye(qubit);
O = kron(Z, I);

%% history variable storage
loss_hist=[];
output_hist=[];
sample_hist=[];
%% Training loop
iter = 1;
for epoch = 1:10
% VQC Forward pass (Variational Quantumn Classify)
    for i = 1:length(x1_normalize)

        % Variable change -> build a variation circuit
        RYt1 = rotation_matrix(theta1);
        RYt2 = rotation_matrix(theta2);
        W = kron(RYt1, RYt2);

        % quantum state preparation
        psi = W * U{i} * state0;

        % measurement
        % psi' stands for conjugate transpose
        % psi' is bra vector (row vector) | psi is ket vector (column vector)
        f = real(psi' * O * psi); % expectation value -> can be complex -> so we use real to take the real part only
        % loss
        y = label(i);
        loss = (f - y)^2;

        %% store history
        % the output is in the boundary [-1;1]
        % easily recognize the quantum vaue
        loss_hist(iter) = loss;
        output_hist(iter) = f;
        sample_hist(iter) = iter;

        disp(['Sample ', num2str(i), ' f = ', num2str(f), ...
              ' loss = ', num2str(loss)])
        
        theta1_old = theta1;
        theta2_old = theta2;
        % gradient updated
        theta1 = shiftrule(theta1_old, theta2_old, state0, U{i}, O, eta, 1);
        theta2 = shiftrule(theta2_old, theta1_old, state0, U{i}, O, eta, 0);
        iter = iter + 1;
    end
end

%% create the result table
results_table = table(sample_hist', output_hist', loss_hist', ...
    'VariableNames', {'Iteration','Output','Loss'});

disp(results_table)

%% plot loss of the training
figure;
plot(sample_hist, loss_hist,'LineWidth',2)
xlabel('Iteration')
ylabel('Loss')
title('Training Loss Curve')
grid on

%% plot the output
figure;
plot(sample_hist, output_hist,'LineWidth',2)
xlabel('Iteration')
ylabel('Output')
title('Model Output Curve')
grid on

%% saving model
theta1_final = theta1;
theta2_final = theta2;

save('vqc_model.mat','theta1_final','theta2_final','max_val','state0','O')
%% Parameter-shift rule to compute gradient
function theta_new = shiftrule(theta, other_theta, state0, U_i, O, eta, isTheta1)

    if isTheta1 == 1
        Wp = kron(rotation_matrix(theta + pi/2), rotation_matrix(other_theta));
        Wm = kron(rotation_matrix(theta - pi/2), rotation_matrix(other_theta));
    else
        Wp = kron(rotation_matrix(other_theta), rotation_matrix(theta + pi/2));
        Wm = kron(rotation_matrix(other_theta), rotation_matrix(theta - pi/2));
    end

    psi_p = Wp * U_i * state0;
    psi_m = Wm * U_i * state0;

    f_p = psi_p' * O * psi_p;
    f_m = psi_m' * O * psi_m;

    grad = (f_p - f_m)/2;

    theta_new = theta - eta * grad;
end

function RY = rotation_matrix(val)
    RY = [cos(val/2),-sin(val/2);sin(val/2),cos(val/2)];
end