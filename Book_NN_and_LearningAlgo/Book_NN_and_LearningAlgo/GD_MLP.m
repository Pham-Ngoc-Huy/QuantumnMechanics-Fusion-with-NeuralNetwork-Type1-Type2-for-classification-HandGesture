function [w1,w2]=GD_MLP(e,eta,u,o1,net1,net2,w1,w2)

% derivative bipolar sigmoid
df1=@(x) 2*exp(-x)./(1+exp(-x)).^2;
df2=@(x) 2*exp(-x)./(1+exp(-x)).^2;

delta2 = e.*df2(net2);
delta1 = (w2(2:end,:) * delta2) .* df1(net1);

rate2 = -o1 * delta2';
rate1 = -u  * delta1';

w2 = GD(w2,eta,rate2);
w1 = GD(w1,eta,rate1);

end