function [vis_im_st] = comp_stack(u1,u2)

if size(u1) ~= size(u2)
    error('images not the same size')
end


switch ndims(u1)
  case 2
    vis_im_st = zeros([size(u1) 1 2]);
    vis_im_st(:,:,1,1) = u1;
    vis_im_st(:,:,1,2) = u2;
  case 3
    vis_im_st = zeros([size(u1) 2]);
    vis_im_st(:,:,:,1) = u1;
    vis_im_st(:,:,:,2) = u2;
end

    