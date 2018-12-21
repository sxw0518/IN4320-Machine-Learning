function phi = rbf(s,rho)
c =[1,2,3,4,5,6];
phi = zeros(6,1);
eta = 1/(rho^2);
for i = 1:6
    phi(i) = exp(-eta*(s-c(i))^2);
end
phi = phi./sum(phi);
end