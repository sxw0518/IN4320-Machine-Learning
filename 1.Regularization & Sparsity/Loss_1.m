function [x] = Loss_1( lambda, r)
x = ((-1-r).^2 + (1-r).^2)*0.5 + lambda* abs(1-r);
end