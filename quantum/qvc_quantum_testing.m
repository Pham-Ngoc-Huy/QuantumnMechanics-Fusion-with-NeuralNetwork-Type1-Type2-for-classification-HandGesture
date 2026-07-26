clc;
clear;

%% load model has trained
load('vqc_model.mat')

%% load data test
file_name = 'data_qc.csv';
dataset = readtable(file_name);
x1_test = dataset.x1;
x2_test = dataset.x2;
label_test = dataset.label;

%% normalize giống training
x1_test_normalize = (x1_test/max_val) * pi;
x2_test_normalize = (x2_test/max_val) * pi;

%% observable size
qubit = 2;
Z = [1,0;0,-1];
I = eye(qubit);
O = kron(Z,I);

%% prediction storage
pred = zeros(length(x1_test_normalize),1);
output_val = zeros(length(x1_test_normalize),1);

%% loop test
for i = 1:length(x1_test_normalize)

    % encoding
    RY1 = rotation_matrix(x1_test_normalize(i));
    RY2 = rotation_matrix(x2_test_normalize(i));
    U = kron(RY1,RY2);

    % variational circuit
    RYt1 = rotation_matrix(theta1_final);
    RYt2 = rotation_matrix(theta2_final);
    W = kron(RYt1,RYt2);

    % forward
    psi = W * U * state0;

    % expectation
    f = real(psi' * O * psi);

    output_val(i) = f;

    % classification
    if f >= 0
        pred(i) = 1;
    else
        pred(i) = -1;
    end
end

%% result table
result_table = table(x1_test, x2_test, label_test, pred, output_val,...
    'VariableNames',{'x1','x2','TrueLabel','Predicted','Output'});

disp(result_table)

%% accuracy
accuracy = sum(pred == label_test)/length(label_test);
disp(['Accuracy = ', num2str(accuracy)])

%% plot comparison
figure;
plot(label_test,'o-','LineWidth',2)
hold on
plot(pred,'x-','LineWidth',2)
legend('True','Predicted')
title('Prediction Comparison')
grid on

%% rotation matrix
function RY = rotation_matrix(val)
    RY = [cos(val/2),-sin(val/2);sin(val/2),cos(val/2)];
end