function [cdfmap,thresmap] = a_contrario_cdf(simmap,param)
% function [cdfmap,thresmap] = a_contrario_cdf(simmap,param)
% This function computes the a contrario detection given a
% similarity map.
% Input arguments:
%       -simmap: a 2D similarity map. Usually it will be the
%       output ofthe sim_function.
%       -param: a structure which fields are the following:
%             -param.func_psim: a string corresponding to
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
%             -param.spot: a matrix the same size as simmap.
% 	      It contains the spot which is used to build
% 	      the underlying model in the a contrario
% 	      method.
% 	      -param.spotsize: an integer corresponding to
% 	      the size of the support of the spot.
% 	      -param.min: if 'y' is chosen then the a
% 	      contrario method is computed on the local
% 	      minima. If 'n' is chosen then it is computed
% 	      on the whole image.
% 	      -param.thres: a real number, the threshold
% 	      used in the a contrario method. It is the
% 	      same as param_detect.thres in
% 	      a_contrario_detection.
% 	      -param.template: a 2D matrix. The template
% 	      used to compute simmap in case of a template
% 	      matching.
%             -param.psize: an integer, the size of the 
%             patch.
% Output arguments:
%        -cdfmap: the c.d.f P(event happens in a contrario
%        model).
%        -thresmap: the thresholded cdfmap (up to
%        param.thres). It is binary matrix. One corresponds
%        to "the offset is detected".
% See also: sim_function, minmap, a_contrario_detection, 
% empirical_comp.    
%       
% Valentin De Bortoli (01/11/2017)    
func_psim = param.func_psim;
matching_psim = param.matching_psim;
h = param.spot;
spotsize = param.spotsize;

%similarity map

switch func_psim
  case {'ps','cos'}
    simmap = -simmap;
end

%use local minima or not

if strcmp(param.min ,'y')
    input = minmap(simmap);
else
    input = simmap;
end

implementation_bool = strcmp(func_psim,'Linfinite') | strcmp(func_psim,'L1') | ...
    (strcmp(matching_psim, 'template') & strcmp(func_psim, 'L2'));

if implementation_bool == true %check if method is implemented
    error('no implementation (yet!)')
elseif strcmp(matching_psim, 'internal') & strcmp(func_psim, 'L2') 
    %use a_contrario_detection.m
    param_detect.input = 'map';
    param_detect.spot = h;    
    param_detect.fixed = 'position';
    param_detect.px = 1;
    param_detect.py = 1;
    param_detect.Lp = param.psize;
    param_detect.thres = param.thres;
    param_detect.spot_option = 'ADSN';
    param_detect.eig_option = 'approx_frob';
    param_detect.inv_option = 'direct';
    param_detect.cdf_option = 'woodf';
    param_detect.fill_method = 'offset_patch';
    [thresmap, eig_m, cdfmap] = a_contrario_detection(input,param_detect);
    %version présente dans patch_sim
else
    h = param.spot;
    L = size(h,1);
    auto_h = real(ifft2(abs(fft2(h)).^2))/L^2;
    snorm_auto_h = sum(sum(auto_h.^2));
    switch matching_psim
      case 'internal'
        auto_hZ = zeros(size(auto_h)); %autocorrelation in Z
        auto_hZ(1:L-spotsize,1:L-spotsize) = auto_h(1:L-spotsize,1:L-spotsize);
        snorm_auto_hZ = sum(sum(auto_hZ.^2));
        p = param.psize;
        mask = zeros(L);
        mask(1:p,1:p) = 1;
        auto_mask_nor = real(ifft2(abs(fft2(mask)).^2)/(p^2));
        m = auto_h * p^2;
        %        var = 2*(snorm_auto_hZ) - auto_hZ.^2 - auto_hZ(1,1)^2;
        var = sum(sum(auto_mask_nor.*auto_h.^2));
        var = var*p^2;  
        %var = var*p^2; %not true variance complicated term...
        if strcmp(func_psim,'cos') %cosine is obtained thanks to ps
            factor = p^2*auto_h(1,1);
            m = m/factor;
            var = var/(factor^2);
        elseif strcmp(func_psim, 'L2asymp')
            m = 2*p^2*(auto_h(1,1) - auto_h);
            var =  4*sum(sum(auto_mask_nor.*(auto_h.^2)))+2* ...
                   real(ifft2(fft2(auto_mask_nor).*fft2(auto_h.^2)))-8* ...
                   real(ifft2(fft2(auto_h.*auto_mask_nor).*fft2(auto_h)));%terme
                                                                          %à
                                                                          %rajouter
                                                                          %2*A(A(h))(2t)
            var = 2*var;
            var = var*p^2;
        end
      case 'template'
        X0 = param.template;
        p = size(X0,1);
        maskX0 = zeros(size(h));
        maskX0(1:p,1:p) = X0;        
        var = sum(sum(maskX0 .* real(ifft2(fft2(maskX0).* ...
                                           fft2(auto_h)))));
        if strcmp(func_psim,'cos')
            var = var/auto_h(1,1);
        end
        m=0;
    end
    % compute cdf
    cdfmap = normcdf(input,m,sqrt(var));
    thresmap = (cdfmap<param.thres).*1;
end
