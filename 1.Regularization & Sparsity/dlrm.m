function [x] = dlrm(mm,b1,mp,lambda)
x = mm.*2 - 2* b1 + lambda*sign(mm-mp);
end