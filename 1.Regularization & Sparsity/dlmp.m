function [x] = dlmp(mp,b0,mm,lambda)
x = mp.*2 - 2/554* b0 - lambda*sign(mm-mp);
end