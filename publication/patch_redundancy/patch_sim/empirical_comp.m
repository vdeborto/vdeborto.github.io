function [] = empirical_comp(empirical_psim_m,param)
% function [] = empirical_comp(empirical_psim_m,param)
% This function produces a comparison between experimental
% values for the different similarity functions and matching
% settings. It plots the different histogram of empirical
% similarity functions for different matching and compare
% them to the expected p.d.f. In the case of the L2
% similarity function no histogram is produced but c.d.f are
% compared. We also show the offset correlation matrix.
% Input arguments:
%       -empirical_psim_m: a matrix of size nfunc x nmatching
%       x number of pixels of the original image. nfunc is the
%       number of simialrity functions tested and nmatching is
%       the number of matching setting tested. It is usually
%       the ouput of the empirical_psim.m function.
%       -param: a structure which field entries are the
%       following:
% 	-param.func_psim: a string corresponding to
%       the similarity function chosen. It must
% 	obviously corresponds to the similarity
% 	function which was used to build simmap.
% 	Options are 'L2', 'L1', 'Linfinite', 'ps' and
% 	'cos'.
% 	-param.matching_psim: a string correspond to
% 	the matching chosen. It must obviously
% 	corresponds to the matching which was used to
% 	build simmap. Options are 'internal' and
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
% 	-param.name: a string, the name used to save all
% 	the images.
% Output arguments: NONE
% See also: empirical_psim, a_contrario_cdf, compare_gaussian
% Remarks: this function produces eps files in ./res. Such a
% folder must exist. A typical file is:
% 'matchingmethod_similarityfunction_name_patchsize.eps'

func_psim_c = param.func_psim_c;
matching_psim_c = param.matching_psim_c;
h = param.spot;
Nfunc = length(func_psim_c);
Nmatching = size(matching_psim_c,2);

L = size(h,1);
auto_h = real(ifft2(abs(fft2(h)).^2))/L^2;
% if L<2*param.spotsize
%     error('image not large enough')
% end
%auto_h_Z = auto_h(1:L-param.spotsize,1:L-param.spotsize);
% autocorrelation on Z

for nmatching = 1:Nmatching
    
    switch matching_psim_c{nmatching}
      case 'internal'
        p = param.psize;
        t = param.poffset;
        mask = zeros(L);
        mask(1:p,1:p) = 1;
        auto_mask_nor = real(ifft2(abs(fft2(mask)).^2)/(p^2));
        for nfunc = 1:Nfunc
            vec_raw = squeeze(empirical_psim_m(nfunc,nmatching,:));
            switch func_psim_c{nfunc}
              case 'L2'
                ft = offset_correlation_function(auto_h,t(1)-1,t(2)-1, ...
                                                 'patch_pixel_pos');
                mat = func2mat(ft,p); 
                figure; imshow(mat,[])
                imwrite(new_range(mat), ['./res/', param.name, '_offset_mat.png'])
                eigL2 = real(eig(mat));
                if max(abs(imag(eigL2)))>10^-8
                    error('imaginary part non zero')
                end
                figure; plot(1:length(eigL2),sort(eigL2,'descend'),'*-')
                vec = vec_raw;
                [out1,x] = ecdf(vec);
                x = x';
                out1 = out1';
                out2 = cdf_wchi(repmat(eigL2,1,length(x)),'direct',x, ...
                                                'imhof');
                pause(1)
                figure;
                plot(x,out1) %empirical cdf
                hold on
                plot(x,out2) %expected cdf
                name = ['./res/internal_l2_', param.name, '_', num2str(p)];
                export_fig(name, '-transparent', '-eps');
              case 'ps'
                vec = vec_raw - p^2*auto_h(t(1),t(2));
                auto_ht1 = circshift(circshift(auto_h,t(1)-1,1),t(2)-1,2);
                auto_ht2 = circshift(circshift(auto_h,-t(1)+1,1),-t(2)+1,2);
                prodt = auto_ht1 .* auto_ht2;
                % var = 2*(sum(sum(auto_h_Z.^2)) + sum(sum(prodt)));
                % var = var - auto_h_Z(1,1)^2 - auto_h_Z(t(1),t(2))^2;
                var = sum(sum((auto_h.^2 + prodt).*auto_mask_nor));
                vec = vec / sqrt(p^2*var);
                name = ['./res/internal_ps_', param.name, '_', num2str(p)];
                compare_gaussian(vec,param,name)
              case 'cosine'
                vec = vec_raw - auto_h(t(1),t(2))/auto_h(1,1);
                auto_ht1 = circshift(circshift(auto_h,t(1)-1,1),t(2)-1,2);
                auto_ht2 = circshift(circshift(auto_h,-t(1)+1,1),-t(2)+1,2);
                prodt = auto_ht1 .* auto_ht2;
                % var = 2*(sum(sum(auto_h_Z.^2)) + sum(sum(prodt)));
                % var = var - auto_h_Z(1,1)^2 - auto_h_Z(t(1),t(2))^2;
                var = sum(sum((auto_h.^2 + prodt).* ...
                              auto_mask_nor));
                var = var/(auto_h(1,1)^2);
                vec = vec / sqrt(var/(p^2));
                name = ['./res/internal_ps_', param.name, '_', num2str(p)];
                compare_gaussian(vec,param,name)
              case 'Linfinite'          
                fprintf('no implementation \n')                        
              case 'L1'
                fprintf('no implementation \n')
              case 'L2asymp'
                vec = vec_raw - 2*p^2*(auto_h(1,1) - auto_h(t(1),t(2)));
                auto_ht1 = circshift(circshift(auto_h,t(1)-1,1),t(2)-1,2);
                auto_ht2 = circshift(circshift(auto_h,-t(1)+1,1),-t(2)+1, ...
                                       2);
                var = 2*sum(sum(((2*auto_h - auto_ht1 - auto_ht2).^2).*auto_mask_nor));
                vec = vec / sqrt(var*p^2);
                compare_gaussian(vec,param,name)
              end
        end                            
      case 'template'
        X0 = param.template;
        p = size(X0,1);
        maskX0 = zeros(L);
        maskX0(1:p,1:p) = X0;
        name = [param.name, '_', num2str(p)];
        for nfunc = 1:Nfunc
            vec_raw = empirical_psim_m(nfunc,nmatching,:);
            switch func_psim_c{nfunc}
              case 'L2'
                fprintf('no implementation \n')
              case 'ps'
                var = sum(sum(maskX0 .* real(ifft2(fft2(maskX0).* ...
                                                   fft2(auto_h)))));
                vec = vec_raw/sqrt(var);
                name = ['./res/template_ps_', param.name, '_', num2str(p)];
                compare_gaussian(vec,param,name)
              case 'cosine'
                var = sum(sum(maskX0 .* real(ifft2(fft2(maskX0).* ...
                                                   fft2(auto_h)))));
                var = var/auto_h(1,1);
                vec = vec_raw*sqrt(p^2/var);
                name = ['./res/template_cos_', param.name, '_', num2str(p)];
                compare_gaussian(vec,param,name)
              case 'Linfinite'
                fprintf('no implementation \n')                        
              case 'L1'
                fprintf('no implementation \n')
              case 'L2asymp'
                fprintf('no implementation \n')
            end
        end
    end
end