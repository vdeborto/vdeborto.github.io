%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %                 SIMILARITY DETECTION                 %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
imtest = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_06.png'])));
imtest = imtest - mean(imtest(:));
L = size(imtest,1);
p = 20;
sigma_t = [0 1 10 20 50];
spotsize_t = [1 5 10 20 100]; %
%nfa_t = [1 5 10 100 1000]; 
nfa_t = 1;
method_c{1} = 'L2asymp';
method_c{2} = 'ps';
method_c{3} = 'cos';
%method_c{4} = 'L2';
minimum_c{1} = 'n';
%minimum_c{2} = 'n';

size_t = [length(sigma_t) length(spotsize_t) length(nfa_t) length(method_c) ...
          length(minimum_c)];
N = prod(size_t)

for n = 1:N
    n
    [n1,n2,n3,n4,n5] = ind2sub(size_t, n);
    % adding noise
    sigma = sigma_t(n1);
    im = imtest + sqrt(sigma) * randn(L,L);
    % spot choice
    L_h = spotsize_t(n2);
    h = im(1:L_h,1:L_h);
    if L_h>1
        h = h - mean(h(:));
        h = h * std(im(:)) / (std(h(:))*L_h);
    else
        h = std(im(:));
    end
    
    % show one image 
    if n == sub2ind(size_t, 1, 3, 1, 1, 1)
        vpv(imtest, adsn_generative(1,h,L,'Z'))
    end
    % nfa choice
    nfa = nfa_t(n3);
    % method choice
    method = method_c{n4};
    % minimum selection 
    min_selection = minimum_c{n5};
    
    [m,var] = internal_similarity_stat(h,method,p,L);
    simmap = internal_similarity_map(im,method,p);
    if strcmp(min_selection,'n')
        [cdfmap,thresmap] = cdf_map(simmap, m, var, p, method, nfa);
    else 
        [cdfmap,thresmap] = cdf_map(minmap(simmap), m, var, p, method, ...
                                    nfa);
    end
    cdfmap_c{n1,n2,n3,n4,n5} = cdfmap;
    thresmap_c{n1,n2,n3,n4,n5} = thresmap;    
    simmap_c{n1,n2,n3,n4,n5} = simmap;        
end
toc;

% we save the quantity cdfmap_c{n1,n2,n3,n4,n5}
% as a matrix of size nsigma x nspot x nmethod
cdfmap_st = zeros(L,L,length(sigma_t),length(spotsize_t),length(method_c));
cdfmapmin_st = zeros(L,L,length(sigma_t),length(spotsize_t),length(method_c));
for n1 = 1:length(sigma_t)
    for n2 = 1:length(spotsize_t)
        for n3 = 1:length(method_c)
            cdfmap_st(:,:,n1,n2,n3) = cdfmap_c{n1,n2,1,n3,1};
            cdfmapmin_st(:,:,n1,n2,n3) = minmap(cdfmap_c{n1,n2,1,n3,1});            
        end
    end
end
thresmap_st = 1.*cdfmap_st<(nfa/(L-p)^2);
thresmapmin_st = 1.*cdfmapmin_st<(nfa/(L-p)^2);
% vpv(dim_3Dstack(thresmap_st(:,:,1,:,1)), dim_3Dstack(thresmap_st(:,:,2,:,1)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,3,:,1)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,4,:,1)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,5,:,1)), ...
%                                                   'nw nc', dim_3Dstack(thresmap_st(:,:,1,:,2)), dim_3Dstack(thresmap_st(:,:,2,:,2)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,3,:,2)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,4,:,2)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,5,:,2)), ...
%                                                       'nw nc', dim_3Dstack(thresmap_st(:,:,1,:,3)), dim_3Dstack(thresmap_st(:,:,2,:,3)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,3,:,3)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,4,:,3)), ...
%                                                   dim_3Dstack(thresmap_st(:,:,5,:,3)))
% vpv(dim_3Dstack(thresmapmin_st(:,:,1,:,1)), dim_3Dstack(thresmapmin_st(:,:,2,:,1)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,3,:,1)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,4,:,1)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,5,:,1)), ...
%                                                   'nw nc', dim_3Dstack(thresmapmin_st(:,:,1,:,2)), dim_3Dstack(thresmapmin_st(:,:,2,:,2)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,3,:,2)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,4,:,2)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,5,:,2)), ...
%                                                       'nw nc', dim_3Dstack(thresmapmin_st(:,:,1,:,3)), dim_3Dstack(thresmapmin_st(:,:,2,:,3)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,3,:,3)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,4,:,3)), ...
%                                                   dim_3Dstack(thresmapmin_st(:,:,5,:,3)))

% we save the results in ~/patch_sim/res/res_internal_percep
nfa_t = [1 10 50];
img_ori = draw_patch(repmat(imtest,1,1,3),1,1,p,'draw','green');
for n1 = 1:length(sigma_t)
    for n2 = 1:length(spotsize_t)
        for n3 = 1:length(method_c)
            sigma_s = ind2str(sigma_t(n1),max(sigma_t));
            spotsize_s = ind2str(spotsize_t(n2),max(spotsize_t));
            method_s = method_c{n3};
            name = [method_s '_' spotsize_s '_' sigma_s '.png'];
            img = double(squeeze(cdfmap_st(:,:,n1,n2,n3)));
            imwrite(img,['cdf_' name])
            for n4 = 1:length(nfa_t)
                n4
                nfa = nfa_t(n4);
                nfa_s = ind2str(nfa_t(n4), max(nfa_t));
                name_nfa = [nfa_s '_' name];
                thresmap = 1.*img<(nfa/(L-p)^2);
                thresmapmin = 1.*squeeze(cdfmapmin_st(:,:,n1,n2,n3))<(nfa/(L-p)^2);
                imwrite(thresmap,['thres_' name_nfa])
                imwrite(thresmapmin,['minthres_' name_nfa])
                
                [px,py] = find(thresmap>0);
                [pxm,pym] = find(thresmapmin>0);
                percepthres = img_ori;
                percepthresmin = img_ori;
                for k1 = 1:length(px)
                    percepthres = draw_patch(percepthres,px(k1),py(k1),p, ...
                                             'draw');
                end
                for k2 = 1:length(pxm)
                    percepthresmin = draw_patch(percepthresmin,pxm(k2),pym(k2),p, ...
                                             'draw');
                end
                imwrite(percepthres,['percep_thres_' name_nfa])
                imwrite(percepthresmin,['percep_minthres_' name_nfa])

            end
        end
    end
end    