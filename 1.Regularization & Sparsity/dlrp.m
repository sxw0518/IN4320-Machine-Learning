function [x] = dlrp(mp,b0,mm,lambda)
x = mp.*2 - 2* b0 - lambda*sign(mm-mp);
end