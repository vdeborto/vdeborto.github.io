function [im_st] = adsn_generative(n,h,L,mode)

L_h = size(h,1);
if L_h>L
    error('size spot greater than L')
end
switch mode
  case 'periodic'
    h_L = zeros(L);
    h_L(1:L_h,1:L_h) = h;
    im_st = randn(L,L,n);
    im_st = real(ifft2(fft2(im_st) .* fft2(h_L)));
  case 'Z'
    h_L = zeros(L+L_h);
    h_L(1:L_h,1:L_h) = h;
    im_st_int = randn(L+L_h,L+L_h,n);
    im_st_int = real(ifft2(fft2(im_st_int) .* fft2(h_L)));
    im_st = im_st_int(1:L,1:L,:);
end
