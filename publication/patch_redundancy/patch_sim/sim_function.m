function [sim] = sim_function(I,param)
% function [sim] = sim_function(I,param)
% This function computes the similarity function in a fixed
% matching setting for one image. In the case of internal
% matching the output is a matrix the size of the original
% image and the (x,y) entry contain the similarity function
% evaluated between the upper-left patch and the patch with
% upper-left coordinate at (x,y). In the template matching
% case the the output evaluated at (x,y) contains the value
% of the similarity function for patch in I with upper-left
% coordinate (x,y) and the template.
% Input arguments:
%       -I: a 2D matrix corresponding to a grey image;
%       -param: a structure containing the different
%       parameters:
% 	-param.func_psim: a string corresponding to
% 	      the similarity function chosen. It must
% 	      obviously corresponds to the similarity
% 	      function which was used to build simmap.
% 	      Options are 'L2', 'L1', 'Linfinite', 'ps' and
% 	      'cos'.
% 	      -param.matching_psim: a string correspond to
% 	      the matching chosen. It must obviously
% 	      corresponds to the matching which was used to
% 	      build simmap. Options are 'internal' and
% 	      'template'.
% 	      -param.template: a 2D matrix. The template
% 	      used to compute simmap in case of a template
% 	      matching.
%             -param.psize: an integer, the size of the 
%             patch.
% Output arguments:
%        -sim: a 2D matrix of the same size of I containing
%        the different values of the similarity function.
% Remark: this function is used to produce simmap in
% a_contrario_cdf and is also in the perceptual study,
% perceptual_script. IT SHOULD NOT BE USED in an
% empirical study to show the validity of the hypotheses.
% empirical_psim computes for one offset but different
% similarity functions and different matching settings the
% same quantity. It is used in empirical_comp and
% empirical_script.
matching_psim = param.matching_psim;
func_psim = param.func_psim;
mask = zeros(size(I));
L = size(I,1);
switch matching_psim
  case 'internal'
  p = param.psize;
  mask(1:p,1:p) = 1;
    switch func_psim
      case {'L2','L2asymp'}
        sim = sum(sum(I.^2.*mask));
        sim = sim + ifft2(conj(fft2(mask)) .* fft2(I.^2));
        sim = sim - 2*ifft2(conj(fft2(mask.*I)) .* fft2(I));
        %sim = sqrt(sim);    
      case 'cos'
        sim = ifft2(conj(fft2(mask.*I)) .* fft2(I));
        sim = sim ./ sqrt(ifft2(fft2(I.^2).*conj(fft2(mask))) .* sum(sum(I.^2.* ...
                                                          mask)));
        %    sim = -sim;
      case 'ps'
        sim = ifft2(conj(fft2(mask.*I)) .* fft2(I));
        %sim=-sim;
      case 'Linfinite'
        patch = patch_extract(I,1,1,p);
        for t = 1:length(I(:))
            [tx,ty] = ind2sub(size(I), t);
            patcht = patch_extract(I,tx,ty,p);
            sim(tx,ty) = max(abs(patcht(:) - patch(:)));
        end
      case 'L1'
        patch = patch_extract(I,1,1,p);
        for t = 1:length(I(:))
            [tx,ty] = ind2sub(size(I), t);
            patcht = patch_extract(I,tx,ty,p);
            sim(tx,ty) = sum(abs(patcht(:) - patch(:)));
        end
    end
  case 'template'
    X0 = param.template;
    p = size(X0,1);
    mask(1:p,1:p) = 1;
    maskX0 = zeros(L);
    maskX0(1:p,1:p) = X0;
    switch func_psim
      case 'L2'
        sim = sum(sum(X0.^2));
        sim = sim + ifft2(conj(fft2(mask)) .* fft2(I.^2));
        sim = sim - 2*ifft2(conj(fft2(maskX0)) .* fft2(I));
        sim = sqrt(real(sim));
      case 'cos'
        sim = ifft2(conj(fft2(maskX0)) .* fft2(I));
        sim = sim ./ sqrt(ifft2(fft2(I.^2).*conj(fft2(mask))) .* sum(sum(X0.^2.* ...
                                                          mask)));
      case 'ps'
        sim = ifft2(conj(fft2(maskX0)) .* fft2(I));
      case 'Linfinite'
        patch = X0;
        for t = 1:length(I(:))
            [tx,ty] = ind2sub(size(I), t);
            patcht = patch_extract(I,tx,ty,p);
            sim(tx,ty) = max(abs(patcht(:) - patch(:)));
        end
      case 'L1'
        patch = X0;
        for t = 1:length(I(:))
            [tx,ty] = ind2sub(size(I), t);
            patcht = patch_extract(I,tx,ty,p);
            sim(tx,ty) = sum(abs(patcht(:) - patch(:)));
        end        
    end
end
sim = real(sim);
