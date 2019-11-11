function [patch] = patch_extract(I, px, py, Lp, position)
%[patch] = patch_extract(I, px, py, Lp, position)
%
%This function extracts a square patch from an image.
%
%Inputs:
%   -I: original image of size Lx x Ly.
%   -px: x-position (up-right) of the patch in I.
%   -py: y-position (up-right) of the patch in I.
%   -Lp: patch size.
%   -position: a string (optional). Default is 'UL'
%    for upper left. Other option is 'center'. In 
%    this case Lp must be odd.
%
%Output:
%   -patch: an image of size Lp x Lp.
if nargin<5
    position = 'UL';
end
[Lx, Ly] = size(I);
pxm = px-1;
pym = py-1;
switch position
  case 'UL'
    patch = I(mod(pxm : pxm+Lp-1, Lx)+1, mod(pym : pym+Lp-1, Ly)+1);
  case 'center'
    Lpm = (Lp-1)/2;
    patch = I(mod(pxm-Lpm : pxm+Lpm, Lx)+1, mod(pym-Lpm : pym+Lpm, Ly)+1);
end

