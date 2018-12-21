function x = lossfunction_4c(rp,rm,a0,a1,lambda)
x = 0;
b = (rp-a0).^2;
c = (rm-a1).^2;
for i = 1:64
    x = b(i)+c(i)+x+lambda*abs(rm(i)-rp(i));
end
end