function [x] = lossfunction3(rp,rm,a0,a1,lambda)
x = 0;
b = zeros(1,64);
c = zeros(1,64);
for i = 1:554
    b = (rp-a0(i,:)).^2 + b;
end
for i = 1:571
    c = (rm-a1(i,:)).^2 + c;
end
for i = 1:64
    x = b(i)+c(i)+x+lambda*abs(rm(i)-rp(i));
end
end