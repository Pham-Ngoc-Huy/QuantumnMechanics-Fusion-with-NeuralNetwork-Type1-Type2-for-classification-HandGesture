clc; clear; close all;

%% Thông số
num_classes = 5;      % số lớp
points_per_class = 1000;  % số điểm mỗi lớp
dim = 2;              % dữ liệu 2D để vẽ

%% Khởi tạo
X = [];
Y = [];

%% Tạo 5 cụm
centers = [ 0 0;
            5 5;
           -5 5;
            5 -5;
           -5 -5];   % 5 tâm khác nhau

for c = 1:num_classes
    data = randn(points_per_class, dim) + centers(c,:);
    X = [X; data];                     % ghép dữ liệu
    Y = [Y; c*ones(points_per_class,1)]; % nhãn lớp
end

%% Plot
figure(1);
hold on;
colors = lines(num_classes);

for c = 1:num_classes
    scatter(X(Y==c,1), X(Y==c,2), 40, ...
            colors(c,:), 'filled');
end

title('5 Clusters Data');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1','Class 2','Class 3','Class 4','Class 5');
grid on;

%% Prepare Data
N = points_per_class;
train_factor = .7*N;
test_factor = N - train_factor;

N_total = size(X,1);
idx = randperm(N_total);

train_factor = round(0.7*N_total);

% Training dataset
trainset = X(idx(1:train_factor),:);
trainlabel = Y(idx(1:train_factor));

% Testing dataset
testset = X(idx(train_factor+1:end),:);
testlabel = Y(idx(train_factor+1:end));

%% NN initial parameters
num_neuron = 10;
w1 = rand(dim+1, num_neuron); % num input x num neurons
w2 = rand(num_neuron+1, 5); % num neurons + 1 x num output

eta = .09;
b1 = 1;
b2 = 1;
o_target = zeros(5,1);

%% Train
for epoch = 1:100
    epoch
    % Train
    E = 0;

    for ii = 1:size(trainset,1)
        x = trainset(ii,:)';
        [o1,o2,net1,net2] = FeedForward_NN(x,b1,b2,w1,w2);
        o_target = zeros(5,1);
        o_target(trainlabel(ii)) = 1;
        e = o_target - o2;
        [w1,w2] = GD_MLP(e,eta,[b1;x],[b2;o1], net1, net2, w1, w2);
        E = E + sum(e.^2);
    end
    E/train_factor
    MSE_Train(epoch) = E/train_factor;

    % Test
    E = 0;
    for ii = 1:size(testset,1)
        x = testset(ii,:)';
        [o1,o2,net1,net2] = FeedForward_NN(x,b1,b2,w1,w2);
        o_target = zeros(5,1);
        o_target(testlabel(ii)) = 1;
        e = o_target - o2;
        E = E + sum(e.^2);
    end
    E/test_factor
    MSE_Test(epoch) = E/test_factor;
end

figure(2)
plot(MSE_Train)
hold on
plot(MSE_Test)


%% ===== Decision Boundary =====
figure(3); hold on;

% Vẽ lại dữ liệu
colors = lines(num_classes);
for c = 1:num_classes
    scatter(X(Y==c,1), X(Y==c,2), 20, colors(c,:), 'filled');
end

% Tạo lưới điểm
x1range = linspace(min(X(:,1))-1, max(X(:,1))+1, 200);
x2range = linspace(min(X(:,2))-1, max(X(:,2))+1, 200);
[x1Grid, x2Grid] = meshgrid(x1range, x2range);

grid_points = [x1Grid(:) x2Grid(:)];

% Dự đoán từng điểm
pred = zeros(size(grid_points,1),1);

for i = 1:size(grid_points,1)
    x = grid_points(i,:)';
    [~,o2,~,~] = FeedForward_NN(x,b1,b2,w1,w2);
    [~,pred(i)] = max(o2);
end

% Reshape lại để vẽ contour
predGrid = reshape(pred, size(x1Grid));

% Vẽ vùng phân lớp
contourf(x1Grid, x2Grid, predGrid, num_classes-1, ...
         'LineColor','none','FaceAlpha',0.2);

title('Decision Boundary of MLP');
xlabel('Feature 1');
ylabel('Feature 2');
grid on;
