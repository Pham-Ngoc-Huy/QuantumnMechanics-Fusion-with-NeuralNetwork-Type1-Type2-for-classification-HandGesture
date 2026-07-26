clear all
clc;

%% Generate Training and Test data
N = 1000;
u = 2*rand(N,1) - 1; % this important -> for y when calculated
% y = zeros(N,1);
y(3,1) = 0; % following the fitness function we just have to use y from t to t-2 -> y(3,1) is oke
train_factor = 0.7*N; % 70% train
test_factor = N - train_factor; % 30% test
%% Construct target objective function -> this function follows the books
for t=4:N
    x1 = y(t-1);
    x2 = y(t-2);
    x3 = y(t-3);
    x4 = u(t-1);
    x5 = u(t-2);
    y(t) = (x1*x2*x3*x5*(x3-1) + x4)/(1+x2^2+x3^2);
end
%% Input data
data = [[0;y(1:end-1)],[0;0;y(1:end-2)],[0;0;0;y(1:end-3)],[0;u(1:end-1)],[0;0;u(1:end-2)]];

nn = randperm(N);
% Training dataset
train_data = data(nn(1:train_factor),:);
target_train = y(nn(1:train_factor));

% Testing dataset
test_data = data(nn(train_factor+1:end),:);
target_test = y(nn(train_factor+1:end));

%% Neural Networks configurations
num_neuron = 4;
w1 = rand(6,num_neuron); % 6x4
w2 = rand(num_neuron+1, 1); % 5x1
eta = 0.005; % learning rate
b1 = 1; % bias of hidden layer
b2 = 1; % bias of output layer

maxepoch = 1000;
%% Train -> with backprogation
for epoch = 1:maxepoch
    epoch
    E = 0;
    nn = randperm(train_factor);
    for ii = nn % stochastic gradient
        x = train_data(ii,:)';
        [o1,o2,net1,net2] = feedforward_nn("regression",x,b1,b2,w1,w2);
        e = target_train(ii) - o2;
        [w1,w2] = GD_MLP("regression",e,eta,[b1;x],[b2;o1], net1, net2, w1, w2);
        E=E+e^2;
    end
    MSE_Train(epoch) = E/train_factor;
    % Test
    E = 0;
    for ii=1:test_factor
        x = test_data(ii,:)';
        [o1,o2,net1,net2] = feedforward_nn("regression",x,1,1,w1,w2);
        e = target_test(ii) - o2;
        E=E+e^2;
        yest(ii)=o2;
    end
    MSE_Test(epoch) = E/test_factor;
end
figure(1)
plot(MSE_Train(1:end), '--b', 'LineWidth',2)
hold on
plot(MSE_Test, '--r', 'LineWidth',2)
E=0;
% using test-the prediction
for ii = 1:test_factor
    x=test_data(ii,:)';
    [o1,o2,net1,net2] = feedforward_nn("regression",x,b1,b2,w1,w2);
    e=target_test(ii) - o2;
    E=E+e^2;
    yest(ii)=o2;
end

figure(2)
% plot
plot(yest, '--b', 'LineWidth',2)
hold on
plot(target_test, '--r', 'LineWidth',2)