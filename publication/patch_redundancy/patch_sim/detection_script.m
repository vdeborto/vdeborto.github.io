%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %                 SIMILARITY DETECTION                 %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imtest = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_01.png'])));
imtest = imtest - mean(imtest(:));
L = size(imtest,1);
p = 20;
t = [10,20];
h = zeros(L);
spotsize = L/16;
spot = imtest(1:spotsize,1:spotsize);
facnorm = sqrt(sum(imtest(:).^2)/sum(spot(:).^2));
spot = facnorm * spot; %normalizing the spot = same variance as the whole image
%spot = ones(1);
%figure; imshow(spot,[])
h(1:spotsize,1:spotsize) = spot;
%h = h-mean(h(:));

template = imtest(1:p,1:p);
%template = template/sqrt(sum(sum(template.^2)));
%func_psim = 'L2asymp';
% func_psim = 'ps';
func_psim = 'L2asymp';
% func_psim = 'Linfinite';
% func_psim = 'L1';
matching_psim = 'internal';
%matching_psim = 'template';

%param.name = name;
param.save = 'on';
param.visible = 'on';
%param.data = Xdata;
param.func_psim = func_psim;
param.matching_psim = matching_psim;
param.template = template;
param.psize = p;
param.poffset = t;
param.spot = h;
param.spotsize = spotsize;
param.min = 'y';
param.thres = 1/L^2;

simmap = sim_function(imtest,param);
[cdfmap,thresmap] = a_contrario_cdf(simmap,param);
% [checkthres,~,checkcdf] = a_contrario_detection(imtest);
% vpv(cdfmap,checkcdf)
vpv(thresmap,minmap(simmap)<Inf)