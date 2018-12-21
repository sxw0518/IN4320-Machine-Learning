function [ x ] = Lossfunction(rp,rm)
x = (2-rp).^2 + (3-rp).^2 + (4-rp).^2 + (-1-rm).^2 + (1-rm).^2 + (0-rm).^2;
end