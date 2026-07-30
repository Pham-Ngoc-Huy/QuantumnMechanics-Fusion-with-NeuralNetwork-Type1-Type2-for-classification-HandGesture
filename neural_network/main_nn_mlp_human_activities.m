clear all
clc;

%% Load data
load("processed_data_movement_classification.mat")

%% Labels
label_participant = all_data(:, 1);

label_activities = all_data(:,2) * 100 + all_data(:,26); % 2 exercises | A: 17 activities / B : 23 activities classes

%% angles (22 columns) -> data has calibrated
cmc1_f = all_data(:,3);
cmc1_a = all_data(:,4);
mcp1 = all_data(:,5);
ip1 = all_data(:,6);
mcp2_f = all_data(:,7);
mcp2_a = all_data(:,8); % this is all NaN remove this out the features
pip2 = all_data(:,9);
mcp3_f = all_data(:,10);
pip3 = all_data(:,11);
mcp4_f = all_data(:,12);
mcp4_a = all_data(:,13);
pip4 = all_data(:,14);
cmc5 = all_data(:,15);
mcp5_f = all_data(:,16);
mcp5_a = all_data(:,17);
pip5 = all_data(:,18);
dip2 = all_data(:,19);
dip3 = all_data(:,20);
dip4 = all_data(:,21);
dip5 = all_data(:,22);
wrist_f  = all_data(:,23);
wrist_a = all_data(:,24);
%% Feature matrix X (22 cols)
X = [
    cmc1_f, ...
    cmc1_a, ...
    mcp1, ...
    ip1, ...
    mcp2_f, ...
    pip2, ...
    mcp3_f, ... 
    pip3, ...
    mcp4_f, ...
    mcp4_a, ...
    pip4, ...
    cmc5, ...
    mcp5_f, ...
    mcp5_a, ...
    pip5, ...
    dip2, ...
    dip3, ...
    dip4, ...
    dip5
    % wrist_f, ...
    % wrist_a
];

fprintf('X: %d rows x %d cols\n', size(X,1), size(X,2));

%% Normalize the data for better gradient purposes
X_min  = min(X);
X_max  = max(X);
X_norm = (X - X_min) ./ (X_max - X_min + 1e-8);


%% turn class from 1 array to dummies -> for example: [1 0 0 0 0 0] -> class 1
classes   = unique(label_activities);
num_class = numel(classes);
fprintf('Activities classes (%d): ', num_class); disp(classes')

Y_onehot = zeros(size(X,1), num_class);
for c = 1:num_class
    Y_onehot(:,c) = (label_activities == classes(c));
end

%% train/test configuration split 70% for training and the remains for testing
N = size(X_norm, 1);
train_factor = round(0.7 * N);
test_factor = N - train_factor;

rng(42); % this the same with random.seed(42) in Python
% since i read library of skitit learn -> see them use  
% Funfact: 42 is a reference from Hitchhikers guide to galaxy book. 
    % The answer to life universe and everything and is meant as a joke. 
    % It has no other significance.

idx_perm = randperm(N); % this randperm make the dataset become more unpredictable
train_idx = idx_perm(1:train_factor);
test_idx = idx_perm(train_factor+1:end);

train_data = X_norm(train_idx, :);
target_train = Y_onehot(train_idx, :);
test_data = X_norm(test_idx,  :);
target_test = Y_onehot(test_idx,  :);

%% architectures
num_input = size(X_norm, 2);   
num_output = num_class;          

hidden_sizes = [128, 64]; % recommend setting when researching  -> can be different but should be descending
num_layers = numel(hidden_sizes);
layer_sizes = [num_input, hidden_sizes];

%% weight initialization
% each weight matrix is (n_in+1) x n_out  — row 1 is the bias weight
w_hidden = cell(num_layers, 1);
for i = 1:num_layers
    n_in  = layer_sizes(i);
    n_out = layer_sizes(i+1);
    limit = sqrt(6 / (n_in + n_out));
    w_hidden{i} = (rand(n_in+1, n_out) * 2 - 1) * limit;
end

n_in_out = hidden_sizes(end);
limit_out = sqrt(6 / (n_in_out + num_output));
w_out = (rand(n_in_out+1, num_output) * 2 - 1) * limit_out;

%% hyper-parameters (place for adjusting)
% Adam default learning rate
eta = 0.001;   
maxepoch = 200;
% smaller batch → better gradient estimates
batch_size = 256;     

% Adam parameters
beta1  = 0.9;
beta2  = 0.999;
eps_ad = 1e-8;

% initialize Adam moment accumulators
m_out = zeros(size(w_out));   v_out = zeros(size(w_out));
m_hid = cellfun(@(w) zeros(size(w)), w_hidden, 'UniformOutput', false);
v_hid = cellfun(@(w) zeros(size(w)), w_hidden, 'UniformOutput', false);
% global step counter
t_adam = 0;   

%% storage - init
loss_train     = zeros(maxepoch, 1);
loss_test      = zeros(maxepoch, 1);
accuracy_train = zeros(maxepoch, 1);
accuracy_test  = zeros(maxepoch, 1);

%% training loop
fprintf('\n%s\n', repmat('=',1,72));
fprintf('  Epoch |  Train Loss | Train Acc |  Test Loss |  Test Acc | Time\n');
fprintf('%s\n', repmat('=',1,72));

for epoch = 1:maxepoch
    tic;
    E_train   = 0;
    correct_train = 0;
    idx = randperm(train_factor);

    %% mini batch SGD with Adam
    for i = 1:batch_size:train_factor
        batch_idx = idx(i : min(i+batch_size-1, train_factor));
        X_b = train_data(batch_idx, :)';   % (51 x batch)
        T_b = target_train(batch_idx, :)'; % (7  x batch)
        n_b = size(X_b, 2);

        %% Forward pass
        [o_hidden, net_hidden, o_output] = feedforward_multihidden(X_b, w_hidden, w_out);
        
        %% Categorical cross-entropy loss  (sum over batch)
        O_clip  = max(o_output, 1e-12);
        E_train = E_train + sum(sum(-T_b .* log(O_clip)));

        %% Batch accuracy
        [~, pred]  = max(o_output, [], 1);
        [~, truth] = max(T_b,      [], 1);
        correct_train = correct_train + sum(pred == truth);

        %% backpropagation -> gradient calculated only
        t_adam = t_adam + 1;
        [grad_w_hidden, grad_w_out] = backprop_multihidden(X_b, T_b, o_hidden, o_output, w_hidden, w_out);

        %% Adam update — output layer
        m_out = beta1 * m_out + (1-beta1) * grad_w_out;
        v_out = beta2 * v_out + (1-beta2) * grad_w_out.^2;
        m_hat = m_out / (1 - beta1^t_adam);
        v_hat = v_out / (1 - beta2^t_adam);
        w_out = w_out - eta * m_hat ./ (sqrt(v_hat) + eps_ad);

        %% Adam update — hidden layers
        for j = 1:num_layers
            m_hid{j} = beta1 * m_hid{j} + (1-beta1) * grad_w_hidden{j};
            v_hid{j} = beta2 * v_hid{j} + (1-beta2) * grad_w_hidden{j}.^2;
            m_hat_j  = m_hid{j} / (1 - beta1^t_adam);
            v_hat_j  = v_hid{j} / (1 - beta2^t_adam);
            w_hidden{j} = w_hidden{j} - eta * m_hat_j ./ (sqrt(v_hat_j) + eps_ad);
        end
    end

    loss_train(epoch)     = E_train / train_factor;
    accuracy_train(epoch) = correct_train / train_factor * 100;

    %% test
    [~, ~, o_test] = feedforward_multihidden(test_data', w_hidden, w_out);

    Tt     = target_test';
    O_clip = max(o_test, 1e-12);
    loss_test(epoch) = sum(sum(-Tt .* log(O_clip))) / test_factor;

    [~, pred_t]  = max(o_test, [], 1);
    [~, truth_t] = max(Tt,     [], 1);
    accuracy_test(epoch) = sum(pred_t == truth_t) / test_factor * 100;

    fprintf('  %4d  |   %.5f   |  %6.2f%%  |  %.5f  |  %6.2f%%  | %.1fs\n', ...
        epoch, loss_train(epoch), accuracy_train(epoch), ...
        loss_test(epoch),  accuracy_test(epoch), toc);
end
%% export the model after training and testing
gesture_model = struct();

% Network architecture
gesture_model.hidden_sizes = hidden_sizes;
gesture_model.num_input = num_input;
gesture_model.num_output = num_output;

% Weights
gesture_model.w_hidden = w_hidden;
gesture_model.w_out = w_out;

% Normalization
gesture_model.X_min = X_min;
gesture_model.X_max = X_max;

% Output classes
gesture_model.classes = classes;

save("gesture_model.mat","gesture_model");

fprintf("\nModel exported to gesture_model.mat\n");
%% plot trend of train and test
figure('Name','Training Curves','NumberTitle','off');

subplot(1,2,1);
plot(1:maxepoch, loss_train, 'b-', 'LineWidth', 1.5); hold on;
plot(1:maxepoch, loss_test,  'r-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss');
title('Cross-Entropy Loss');
legend('Train','Test'); grid on;

subplot(1,2,2);
plot(1:maxepoch, accuracy_train, 'b-', 'LineWidth', 1.5); hold on;
plot(1:maxepoch, accuracy_test,  'r-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Accuracy (%)');
title('Classification Accuracy');
legend('Train','Test'); grid on;


%% confusion matrix only
% Final predictions (train)
[~, ~, o_train_final] = feedforward_multihidden(train_data', w_hidden, w_out);
[~, pred_train_final]  = max(o_train_final, [], 1);
[~, truth_train_final] = max(target_train', [], 1);

% Final predictions (test)
[~, ~, o_test_final]   = feedforward_multihidden(test_data',  w_hidden, w_out);
[~, pred_test_final]   = max(o_test_final, [], 1);
[~, truth_test_final]  = max(target_test', [], 1);

% Class labels as strings (velocity class values)
class_labels = arrayfun(@(c) sprintf('V%g', c), classes, 'UniformOutput', false);
C = confusionmat(truth_train_final, pred_train_final);
C1 = confusionmat(truth_test_final,  pred_test_final);

figure(2)
confusionchart(C)
figure(3)
confusionchart(C1)