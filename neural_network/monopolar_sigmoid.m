% can use other activation functions instead of sigmoid,
% as long as the output is bounded in [0,1].
% choice of function may affect smoothness and gradient behavior. -> hàm kích hoạt phải thỏa là sẽ tăng dần từ 0 -> 1
function output = monopolar_sigmoid(var)
    output = 1./(1+exp(-var));
end
