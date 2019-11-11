function [empirical_psim_m] = empirical_psim(param)
% function [empirical_psim_m] = empirical_psim(param)
% This functions computes the empirical similarity functions
% for multiple entries images, multiple matching settings and
% multiple similarity functions. The output stores these
% functions.
% Input arguments:
%       -param: a structure containing the different
%       parameters of the algorithm:
%       -param.func_psim_c: a cell corresponding to
%       the similarity function chosen. It can contain
%	'L2', 'L1', 'Linfinite', 'ps' 'cos'.
% 	-param.matching_psim_c: a cell corresponding to
% 	the matching chosen. It  'internal' and
% 	'template'.
% 	-param.spot: a matrix the same size as simmap.
% 	It contains the spot which is used to build
% 	the underlying model in the a contrario
% 	method.
% 	-param.spotsize: an integer corresponding to
% 	the size of the support of the spot.
% 	-param.psize: an integer, the size of the patch.
% 	-param.poffset: an integer, the offset used in the
% 	internal mode.
% 	-param.template: a 2D matrix. The template
% 	used to compute simmap in case of a template
% 	matching.
% 	-param.data: a 3D matrix of size size(img) x N where
% 	N is the number images on which we want to compute
% 	the different similarity functions.
% Output arguments:
%        -empirical_psim_m: a matrix of size nfunc x nmatching
%        x N containing the similarity functions for different
%        similarity functions and matching settings.
% Remarks: only one patch per image of param.data is evaluated
% against the template in template mode and only one offset is
% selected in internal mode. This function has only one
% utility: be used in empirical_script in order to confirm
% results. If you want to compute for one image the similarity
% functions for every offset in the internal mode and the
% difference with a template for every patch please use
% sim_function.  
func_psim_c = param.func_psim_c;
matching_psim_c = param.matching_psim_c;
Xdata = param.data;
N = size(Xdata,3);
Nfunc = length(func_psim_c);
Nmatching = length(matching_psim_c);
empirical_psim_m = zeros(Nfunc, Nmatching, N);

for n = 1:N
    X = Xdata(:,:,n);
    for nmatching = 1:Nmatching
        
        switch matching_psim_c{nmatching}
          case 'internal'
            p = param.psize;
            t = param.poffset;
            pX = patch_extract(X,1,1,p);
            pcomp = patch_extract(X,t(1),t(2),p);
          case 'template'
            pcomp = param.template;
            pX = patch_extract(X,1,1,size(pcomp,1));
        end
        
        for nfunc = 1:Nfunc
            switch func_psim_c{nfunc}
              case {'L2','L2asymp'}
                sim_value = sum(sum((pX-pcomp).^2));
              case 'ps'
                sim_value = sum(sum(pX.*pcomp));
              case 'cosine'
                sim_value = sum(sum(pX.*pcomp))/sqrt(((sum(sum(pX.^2)))*(sum(sum(pcomp.^2)))));
              case 'Linfinite'
                sim_value = max(max(abs(pX-pcomp)));
              case 'L1'
                sim_value = sum(sum(abs(pX-pcomp)));
            end
            empirical_psim_m(nfunc,nmatching,n) = sim_value;
        end
        
    end
end
