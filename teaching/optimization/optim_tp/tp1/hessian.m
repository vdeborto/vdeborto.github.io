function M = hessian(A,omega,a,x)

alpha = A * x;
s = sin(omega * x);
c = cos(omega * x);
M = exp(-a * x) * x * [0 -s c ; -s alpha*s -alpha*c ; c -alpha*c -alpha*s];