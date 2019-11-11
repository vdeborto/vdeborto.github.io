function [auto_u] = autocorrelation(u)

auto_u = real(ifft2(abs(fft2(u)).^2));