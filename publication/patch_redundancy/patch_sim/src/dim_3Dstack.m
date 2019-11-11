function [vis_im_st] = dim_3Dstack(im_st)

n = ndims(im_st);
if n == 4
    vis_im_st = im_st;
elseif n ==3
    sz = size(im_st);
    sz_vis = [sz(1:2) 1 sz(3)];
    vis_im_st = reshape(im_st, sz_vis);
else
    error('wrong number of dimensions')
end
    