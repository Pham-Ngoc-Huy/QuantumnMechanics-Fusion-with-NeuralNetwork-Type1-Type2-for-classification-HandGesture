clear all 
clc 

%% Generate Training and Test data
N = 100;
u = 2*rand(N,1)-1; 
y = zeros(N,1); 
train_factor = 0.7*N;
test_factor = N-train_factor;
%% Construct target objective function 
% for t=3:N
%     x1 = y(t-1);
%     x2 = y(t-2);
%     x3 = u(t-1);
%     y(t) = 0.5*x1-0.2*x2+0.1*x1^2+0.3*x3;
% end 

for t=3:N
    x1 = y(t-1);
    x2 = y(t-2);
    x3 = u(t-1);
    y(t) = (0.8 - 0.5*exp(-x1^2))*x1 - 0.3*x2 + 0.2*x3;
end
%% Input data
data = [[0;y(1:end-1)] [0;0;y(1:end-2)] [0;u(1:end-1)]];

nn = randperm(N);
% Training dataset
trainset = data(nn(1:train_factor),:);
trainlabel = y(nn(1:train_factor));

% Testing dataset
testset = data(nn(train_factor+1:end),:);
testlabel = y(nn(train_factor+1:end));

%% Neural Networks configurations
num_neuron = 10;
w1 = rand(4,num_neuron); % 4 x 4
w2 = rand(num_neuron+1, 1); % 5 x 1
eta = .01;
b1 = 1;
b2 = 1;

MaxEpoch = 200;
%% Train
for epoch = 1:MaxEpoch
    epoch
    E = 0;
    nn = randperm(train_factor);
    kk = 1;
    for ii = nn
        x = trainset(ii,:)';
        [o1,o2,net1,net2] = FeedForward_NN(x,b1,b2,w1,w2);
        e = trainlabel(ii) - o2;
        [w1,w2] = GD_MLP(e,eta,[b1;x],[b2;o1], net1, net2, w1, w2);
        E=E+e^2;
    end
    MSE_Train(epoch) = E/train_factor;
    % Test
    E = 0;
    for ii=1:test_factor
        x = testset(ii,:)';
        [o1,o2,net1,net2] = FeedForward_NN(x,1,1,w1,w2);
        e = testlabel(ii) - o2;
        E=E+e^2;
        yest(ii)=o2;
    end
    MSE_Test(epoch) = E/test_factor;
end
%% predict
yttest = zeros(N,1);
for i = 1:N
    x = data(i,:)';
    [~,o2,~,~] = FeedForward_NN(x,1,1,w1,w2);
    yttest(i) = o2;
end
figure(1)
plot(y,'b','LineWidth',2)
hold on 
plot(yttest,'--r','LineWidth',1)


figure(2)
plot(MSE_Train)
hold on
plot(MSE_Test)

disp(MSE_Train)