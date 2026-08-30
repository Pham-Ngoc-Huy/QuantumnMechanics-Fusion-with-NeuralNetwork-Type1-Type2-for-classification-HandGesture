function [updated] = GD(recent,eta,rate)
%GD Summary of this function goes here
%   Detailed explanation goes here
updated = recent - eta*rate;
end
