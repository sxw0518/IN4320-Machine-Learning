function [x] = dlmm(mm,b1,mp,lambda)
x = mm.*2 - 2/571* b1 + lambda*sign(mm-mp);
end