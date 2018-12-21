function [ x ] = ldm( lambda,r )
x=2.*r-lambda*((1-r)./abs(1-r));
x(isnan(x)==1) = 2*r(r==1);
end