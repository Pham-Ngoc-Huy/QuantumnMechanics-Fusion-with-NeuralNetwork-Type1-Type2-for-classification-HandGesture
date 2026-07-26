clear all
clc;

%% load data
load("processed_data_velocity_classification.mat")

%% labels
label_participant = all_data_full(:, 1);
label_velocity    = all_data_full(:, 2);

%% EMG_R (8 cols)
EMG_R_St1_VL  = all_data_full(:, 3);  EMG_R_St2_VL  = all_data_full(:, 4);
EMG_R_St1_BF  = all_data_full(:, 5);  EMG_R_St2_BF  = all_data_full(:, 6);
EMG_R_St1_TA  = all_data_full(:, 7);  EMG_R_St2_TA  = all_data_full(:, 8);
EMG_R_St1_GAL = all_data_full(:, 9);  EMG_R_St2_GAL = all_data_full(:,10);

%% EMG_L (4 cols)
EMG_L_St1_VL  = all_data_full(:,11); EMG_L_St1_BF  = all_data_full(:,12);
EMG_L_St1_TA  = all_data_full(:,13); EMG_L_St1_GAL = all_data_full(:,14);

%% Torque_R (24 cols)
TQ_R_St1_Pelvis_X = all_data_full(:,15); TQ_R_St1_Pelvis_Y = all_data_full(:,16); TQ_R_St1_Pelvis_Z = all_data_full(:,17);
TQ_R_St2_Pelvis_X = all_data_full(:,18); TQ_R_St2_Pelvis_Y = all_data_full(:,19); TQ_R_St2_Pelvis_Z = all_data_full(:,20);
TQ_R_St1_Hip_X    = all_data_full(:,21); TQ_R_St1_Hip_Y    = all_data_full(:,22); TQ_R_St1_Hip_Z    = all_data_full(:,23);
TQ_R_St2_Hip_X    = all_data_full(:,24); TQ_R_St2_Hip_Y    = all_data_full(:,25); TQ_R_St2_Hip_Z    = all_data_full(:,26);
TQ_R_St1_Knee_X   = all_data_full(:,27); TQ_R_St1_Knee_Y   = all_data_full(:,28); TQ_R_St1_Knee_Z   = all_data_full(:,29);
TQ_R_St2_Knee_X   = all_data_full(:,30); TQ_R_St2_Knee_Y   = all_data_full(:,31); TQ_R_St2_Knee_Z   = all_data_full(:,32);
TQ_R_St1_Ankle_X  = all_data_full(:,33); TQ_R_St1_Ankle_Y  = all_data_full(:,34); TQ_R_St1_Ankle_Z  = all_data_full(:,35);
TQ_R_St2_Ankle_X  = all_data_full(:,36); TQ_R_St2_Ankle_Y  = all_data_full(:,37); TQ_R_St2_Ankle_Z  = all_data_full(:,38);

%% Torque_L (9 cols)
TQ_L_St1_Hip_X = all_data_full(:,39); 
TQ_L_St1_Hip_Y = all_data_full(:,40); 
TQ_L_St1_Hip_Z = all_data_full(:,41);
TQ_L_St1_Knee_X = all_data_full(:,42); 
TQ_L_St1_Knee_Y = all_data_full(:,43); 
TQ_L_St1_Knee_Z = all_data_full(:,44);
TQ_L_St1_Ankle_X = all_data_full(:,45); 
TQ_L_St1_Ankle_Y = all_data_full(:,46); 
TQ_L_St1_Ankle_Z = all_data_full(:,47);

%% Metadata (6 cols)
gender = all_data_full(:,48); 
age = all_data_full(:,49);
body_height = all_data_full(:,50); 
body_mass = all_data_full(:,51);
leg_length = all_data_full(:,52); 
foot_length = all_data_full(:,53);

%% Feature matrix X (51 cols)
X = [EMG_R_St1_VL,  EMG_R_St2_VL,  EMG_R_St1_BF,  EMG_R_St2_BF, ...
     EMG_R_St1_TA,  EMG_R_St2_TA,  EMG_R_St1_GAL, EMG_R_St2_GAL, ...
     EMG_L_St1_VL,  EMG_L_St1_BF,  EMG_L_St1_TA,  EMG_L_St1_GAL, ...
     TQ_R_St1_Pelvis_X, TQ_R_St1_Pelvis_Y, TQ_R_St1_Pelvis_Z, ...
     TQ_R_St2_Pelvis_X, TQ_R_St2_Pelvis_Y, TQ_R_St2_Pelvis_Z, ...
     TQ_R_St1_Hip_X,    TQ_R_St1_Hip_Y,    TQ_R_St1_Hip_Z, ...
     TQ_R_St2_Hip_X,    TQ_R_St2_Hip_Y,    TQ_R_St2_Hip_Z, ...
     TQ_R_St1_Knee_X,   TQ_R_St1_Knee_Y,   TQ_R_St1_Knee_Z, ...
     TQ_R_St2_Knee_X,   TQ_R_St2_Knee_Y,   TQ_R_St2_Knee_Z, ...
     TQ_R_St1_Ankle_X,  TQ_R_St1_Ankle_Y,  TQ_R_St1_Ankle_Z, ...
     TQ_R_St2_Ankle_X,  TQ_R_St2_Ankle_Y,  TQ_R_St2_Ankle_Z, ...
     TQ_L_St1_Hip_X,    TQ_L_St1_Hip_Y,    TQ_L_St1_Hip_Z, ...
     TQ_L_St1_Knee_X,   TQ_L_St1_Knee_Y,   TQ_L_St1_Knee_Z, ...
     TQ_L_St1_Ankle_X,  TQ_L_St1_Ankle_Y,  TQ_L_St1_Ankle_Z, ...
     gender, age, body_height, body_mass, leg_length, foot_length];

num_features = size(X, 2);
fprintf('X: %d rows x %d cols\n', size(X,1), num_features);

%% Normalize to [0,1]
X_min = min(X);
X_max = max(X);
X_norm = (X - X_min) ./ (X_max - X_min + 1e-8);  % (N × 51)

%% One-hot encode labels
classes   = unique(label_velocity);
num_class = numel(classes);
fprintf('Velocity classes (%d): ', num_class); disp(classes')

Y_onehot = zeros(size(X,1), num_class);
for c = 1:num_class
    Y_onehot(:,c) = (label_velocity == classes(c));
end

%% train / test split 70/30
N = size(X_norm, 1);
train_factor = round(0.7 * N);
test_factor = N - train_factor;

rng(42);
idx_perm  = randperm(N);
train_idx = idx_perm(1:train_factor);
test_idx  = idx_perm(train_factor+1:end);

train_data   = X_norm(train_idx, :);   % (train_factor × 51)  — rows=samples
target_train = Y_onehot(train_idx, :);
test_data    = X_norm(test_idx,  :);   % (test_factor  × 51)
target_test  = Y_onehot(test_idx,  :);

%% fuzzy parameters
K = 3;
num_rules = num_features * K;
% Precompute feature index for every rule (used in Layers 4 & backprop)
feat_of_rule = ceil((1:num_rules)' / K);         % (153 × 1)

%% Layer 1 parameters: MF centers and widths
C     = repmat(linspace(0.1, 0.9, K), num_features, 1);  % (51 × 3)
Sigma = ones(num_features, K) * 0.3;                      % (51 × 3)

%% Layer 4 parameters: Consequents
lim_conseq = sqrt(6 / (num_rules + num_class));
P = (rand(num_rules, num_class) * 2 - 1) * lim_conseq;  % (153 × 7)
Q = zeros(num_rules, num_class);                          % (153 × 7)

%% MLP parameters
% Input to MLP = defuzzified output = (num_class × batch) = (7 × batch)
hidden_sizes = [64, 32];
num_mlp_layers = numel(hidden_sizes);
mlp_sizes = [num_class, hidden_sizes];   % [7, 64, 32]

w_hidden = cell(num_mlp_layers, 1);
for i = 1:num_mlp_layers
    n_in  = mlp_sizes(i);
    n_out = mlp_sizes(i+1);
    lim   = sqrt(6 / (n_in + n_out));
    w_hidden{i} = (rand(n_in+1, n_out) * 2 - 1) * lim;  % (+1 for bias)
end
lim_out = sqrt(6 / (hidden_sizes(end) + num_class));
w_out   = (rand(hidden_sizes(end)+1, num_class) * 2 - 1) * lim_out;

fprintf('MLP: input=%d → [%s] → output=%d\n', num_class, ...
        num2str(hidden_sizes), num_class);
fprintf('Fuzzy rules: %d features × %d sets = %d rules\n', ...
        num_features, K, num_rules);

%% hyper-parameters
eta = 0.001;
maxepoch   = 300;
batch_size = 512;
beta1 = 0.9;  
beta2 = 0.999;  
eps_ad = 1e-8;

%% adam parameters
m_out = zeros(size(w_out));     
v_out = zeros(size(w_out));
m_hid = cellfun(@(w) zeros(size(w)), w_hidden, 'UniformOutput', false);
v_hid = cellfun(@(w) zeros(size(w)), w_hidden, 'UniformOutput', false);
m_C = zeros(size(C));    
v_C = zeros(size(C));
m_S = zeros(size(Sigma));
v_S = zeros(size(Sigma));
m_P = zeros(size(P));    
v_P = zeros(size(P));
m_Q = zeros(size(Q));    
v_Q = zeros(size(Q));
t_adam = 0;

%% storage
loss_train     = zeros(maxepoch, 1);
loss_test      = zeros(maxepoch, 1);
accuracy_train = zeros(maxepoch, 1);
accuracy_test  = zeros(maxepoch, 1);

fprintf('\n%s\n', repmat('=',1,80));
fprintf('  Epoch |  Train Loss | Train Acc |  Test Loss |  Test Acc | Time\n');
fprintf('%s\n', repmat('=',1,80));

for epoch = 1:maxepoch
    tic;
    E_train = 0;
    correct_train = 0;
    idx_shuf = randperm(train_factor);

    for i = 1:batch_size:train_factor
        batch_idx = idx_shuf(i : min(i+batch_size-1, train_factor));
        X_b = train_data(batch_idx, :)';  
        T_b = target_train(batch_idx, :)'; 
        n_b = size(X_b, 2);

%% Fuzzification
        % Apply Gaussian Form membership
        MU = zeros(num_features, K, n_b);
        for k = 1:K
            diff = X_b - C(:,k);
            MU(:,k,:) = exp( -(diff.^2) ./ (2*Sigma(:,k).^2 + 1e-8) );
        end

%% Rule Firing
        W_fire = reshape(permute(MU, [2,1,3]), num_rules, n_b);  % (153 × batch)

%% Normalization
        S = sum(W_fire, 1) + 1e-8;    % (1 × batch)
        W_norm = W_fire ./ S;              % (153 × batch)

%% Consequent
        X_feat = X_b(feat_of_rule, :);     % (153 × batch)

        F_conseq = bsxfun(@times, reshape(X_feat, num_rules, 1, n_b), ...
                          reshape(P, num_rules, num_class, 1)) + ...
                   reshape(Q, num_rules, num_class, 1);   % (153 × 7 × batch)

        WF = bsxfun(@times, reshape(W_norm, num_rules, 1, n_b), F_conseq);
        % (153 × 7 × batch)

%% Defuzzificationz
        Y_defuzz = reshape(sum(WF, 1), num_class, n_b);   % (7 × batch)

%% Feedforward based on Defuzzification
        %  Y_defuzz (7×batch) → hidden [64→32] → softmax(7)
        [o_hidden, ~, o_output] = feedforward_nn_hidden_01(Y_defuzz, w_hidden, w_out);

%% Loss and Accuracy (Loss Entrioy)
        O_clip  = max(o_output, 1e-12);
        E_train = E_train + sum(sum(-T_b .* log(O_clip)));

        [~, pred]  = max(o_output, [], 1);
        [~, truth] = max(T_b,      [], 1);
        correct_train = correct_train + sum(pred == truth);
%% Backprogation
        t_adam = t_adam + 1;

        [grad_w_hidden, grad_w_out_g, delta_defuzz] = fuzzy_backprop(Y_defuzz, T_b, o_hidden, o_output, w_hidden, w_out);

        delta_WF = repmat(reshape(delta_defuzz, 1, num_class, n_b), num_rules, 1, 1);   % (153 × 7 × batch)


        delta_F = bsxfun(@times, delta_WF, reshape(W_norm, num_rules, 1, n_b));

        delta_Wnorm = reshape(sum(delta_WF .* F_conseq, 2), num_rules, n_b);

        X_feat_3d = reshape(X_feat, num_rules, 1, n_b);
        grad_P = sum(delta_F .* X_feat_3d, 3) / n_b;   % (153 × 7)

        grad_Q = sum(delta_F, 3) / n_b;                 % (153 × 7)

%% normalization
        dot_val     = sum(delta_Wnorm .* W_norm, 1);          % (1 × batch)
        delta_Wfire = (delta_Wnorm - W_norm .* dot_val) ./ S; % (153 × batch)

%% rule firing
        delta_MU = permute(reshape(delta_Wfire, K, num_features, n_b), [2,1,3]);
        % (51 × 3 × batch)

%% fuzzification
        grad_C_out     = zeros(num_features, K);
        grad_Sigma_out = zeros(num_features, K);

        for k = 1:K
            mu_k    = squeeze(MU(:,k,:));          % (51 × batch)
            diff_k  = X_b - C(:,k);               % (51 × batch)
            delt_k  = squeeze(delta_MU(:,k,:));    % (51 × batch)

            sig2 = Sigma(:,k).^2 + 1e-8;          % (51 × 1)
            sig3 = Sigma(:,k).^3 + 1e-8;

            dmu_dC   = mu_k .* diff_k ./ sig2;           % (51 × batch)
            dmu_dSig = mu_k .* (diff_k.^2) ./ sig3;      % (51 × batch)

            grad_C_out(:,k)     = sum(delt_k .* dmu_dC,   2) / n_b;
            grad_Sigma_out(:,k) = sum(delt_k .* dmu_dSig, 2) / n_b;
        end

        % MLP output weights
        m_out = beta1*m_out + (1-beta1)*grad_w_out_g;
        v_out = beta2*v_out + (1-beta2)*grad_w_out_g.^2;
        w_out = w_out - eta*(m_out/(1-beta1^t_adam)) ./ (sqrt(v_out/(1-beta2^t_adam))+eps_ad);

        % MLP hidden weights
        for j = 1:num_mlp_layers
            m_hid{j} = beta1*m_hid{j} + (1-beta1)*grad_w_hidden{j};
            v_hid{j} = beta2*v_hid{j} + (1-beta2)*grad_w_hidden{j}.^2;
            w_hidden{j} = w_hidden{j} - eta*(m_hid{j}/(1-beta1^t_adam)) ./ (sqrt(v_hid{j}/(1-beta2^t_adam))+eps_ad);
        end

        % Fuzzy centers C
        m_C = beta1*m_C + (1-beta1)*grad_C_out;
        v_C = beta2*v_C + (1-beta2)*grad_C_out.^2;
        C   = C - eta*(m_C/(1-beta1^t_adam)) ./ (sqrt(v_C/(1-beta2^t_adam))+eps_ad);

        % Fuzzy widths Sigma
        m_S = beta1*m_S + (1-beta1)*grad_Sigma_out;
        v_S = beta2*v_S + (1-beta2)*grad_Sigma_out.^2;
        Sigma = Sigma - eta*(m_S/(1-beta1^t_adam)) ./ (sqrt(v_S/(1-beta2^t_adam))+eps_ad);
        Sigma = max(Sigma, 0.01);   % keep widths strictly positive

        % Consequent slopes P
        m_P = beta1*m_P + (1-beta1)*grad_P;
        v_P = beta2*v_P + (1-beta2)*grad_P.^2;
        P   = P - eta*(m_P/(1-beta1^t_adam)) ./ (sqrt(v_P/(1-beta2^t_adam))+eps_ad);

        % Consequent intercepts Q
        m_Q = beta1*m_Q + (1-beta1)*grad_Q;
        v_Q = beta2*v_Q + (1-beta2)*grad_Q.^2;
        Q   = Q - eta*(m_Q/(1-beta1^t_adam)) ./ (sqrt(v_Q/(1-beta2^t_adam))+eps_ad);
    end

    loss_train(epoch)     = E_train / train_factor;
    accuracy_train(epoch) = correct_train / train_factor * 100;

%% test evaluation
    Y_dt = fuzzy_forward(test_data', C, Sigma, P, Q, K, num_features, num_class, num_rules, feat_of_rule);
    [~, ~, o_test] = feedforward_nn_hidden_01(Y_dt, w_hidden, w_out);

    Tt     = target_test';
    O_clip = max(o_test, 1e-12);
    loss_test(epoch) = sum(sum(-Tt .* log(O_clip))) / test_factor;

    [~, pred_t]  = max(o_test, [], 1);
    [~, truth_t] = max(Tt, [], 1);
    accuracy_test(epoch) = sum(pred_t == truth_t) / test_factor * 100;

    fprintf('  %4d  |   %.5f   |  %6.2f%%  |  %.5f  |  %6.2f%%  | %.1fs\n', ...
        epoch, loss_train(epoch), accuracy_train(epoch), ...
        loss_test(epoch), accuracy_test(epoch), toc);
end

%% training curve / testing curve
figure('Name','FNN Training Curves','NumberTitle','off');
subplot(1,2,1);
plot(1:maxepoch, loss_train,'b-','LineWidth',1.5); hold on;
plot(1:maxepoch, loss_test, 'r-','LineWidth',1.5);
xlabel('Epoch'); ylabel('Loss'); title('Cross-Entropy Loss');
legend('Train','Test'); grid on;

subplot(1,2,2);
plot(1:maxepoch, accuracy_train,'b-','LineWidth',1.5); hold on;
plot(1:maxepoch, accuracy_test, 'r-','LineWidth',1.5);
xlabel('Epoch'); ylabel('Accuracy (%)'); title('Classification Accuracy');
legend('Train','Test'); grid on;

%% confusion matrix
Y_dtr = fuzzy_forward(train_data', C, Sigma, P, Q, K, num_features, num_class, num_rules, feat_of_rule);
[~,~,o_tr] = feedforward_nn_hidden_01(Y_dtr, w_hidden, w_out);
[~,p_tr] = max(o_tr,[],1);  [~,t_tr] = max(target_train',[],1);

Y_dte = fuzzy_forward(test_data', C, Sigma, P, Q, K, num_features, num_class, num_rules, feat_of_rule);
[~,~,o_te] = feedforward_nn_hidden_01(Y_dte, w_hidden, w_out);
[~,p_te] = max(o_te,[],1);  [~,t_te] = max(target_test',[],1);

figure(2); 
confusionchart(confusionmat(t_tr,p_tr)); 
title('Train Confusion Matrix');
figure(3); 
confusionchart(confusionmat(t_te,p_te)); 
title('Test Confusion Matrix');

%% visualization membership function
feat_names = {'EMG_R_St1_VL','EMG_R_St2_VL','EMG_R_St1_BF','EMG_R_St2_BF'};
set_names  = {'Low','Medium','High'};
colors     = {'b','g','r'};
x_range    = linspace(0,1,300);

figure('Name','Learned Fuzzy MFs','NumberTitle','off');
for f = 1:4
    subplot(2,2,f); hold on;
    for k = 1:K
        mu_plot = exp(-((x_range-C(f,k)).^2)./(2*Sigma(f,k)^2));
        plot(x_range, mu_plot, colors{k},'LineWidth',2,'DisplayName',set_names{k});
    end
    xlabel('Normalised value'); ylabel('\mu');
    title(feat_names{f}); legend; grid on; ylim([0 1.05]);
end
sgtitle('Learned Fuzzy Sets after Training');

fprintf('\nTraining complete.\n');