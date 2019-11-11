%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %                     SCRIPT TEST                      %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath('../')

% Ntest = 1;
% mN = zeros(1,Ntest);
% stdN = zeros(1,Ntest);
% tic;
% for k = 1:Ntest
%     n = 1000;
%     mode = 'Z';
%     h = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
%                         'img_18.png'])));
%     h = h(1:10,1:10);
%     L = 40;
%     im_st = adsn_generative(n,h,L,mode);
%     vis_im_st = dim_3Dstack(im_st);

%     p = 10;
%     method = 'cos';
%     [m,var] = internal_similarity_stat(h,method,p,L);

%     sim_st = internal_similarity_map(im_st,method,p);

%     nfa = 10;

%     [cdf_st,thres_st] = cdf_map(sim_st, m, var, p, method, nfa);
%     %vpv(dim_3Dstack(im_st), 'nc nw', dim_3Dstack(cdf_st), dim_3Dstack(cdf_st))

%     thres_test_st = thres_st(1:L-p,1:L-p,:);
%     sum(thres_test_st(:))/size(thres_st,3);

%     t = [10 20];
%     % mtest = m_c{4,2,3};
%     % vartest = var_c{4,2,3};
%     vec = squeeze(sim_st(t(1),t(2),:)) - m(t(1),t(2));
%     vec = vec/sqrt(var(t(1),t(2)));
%     % vec = vec/sqrt(vartest(t(1),t(2)));
%     % vec = vec/std(vec)
%     param.visible = 'on';
%     param.save = 'on';
%     name = '../res/isgaussian_internal_cos_10_10';
%     compare_gaussian(vec,param,name)
%     %compare_gaussian(vec)

%     sum(thres_test_st(t(1),t(2),:))/size(thres_st,3);
%     mN(k) = abs(mean(vec));
%     stdN(k) = abs(std(vec) - 1);
% end
% toc

%% template

n = 10000;
mode = 'Z';
im_init = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_18.png'])));
h_size = 5;
h = im_init(1:h_size,1:h_size);
template = im_init(1:30,1:30);
L = 128;
im_st = adsn_generative(n,h,L,mode);
vis_im_st = dim_3Dstack(im_st);

p = 90;
method = 'L2asymp';
[m,var] = internal_similarity_stat(h,method,p,L);
sim_st = internal_similarity_map(im_st,method,p);

nfa = 10;

[cdf_st,thres_st] = cdf_map(sim_st, m, var, p, method, nfa);
%vpv(dim_3Dstack(im_st), 'nc nw', dim_3Dstack(cdf_st), dim_3Dstack(cdf_st))

thres_test_st = thres_st(1:L-p,1:L-p,:);
sum(thres_test_st(:))/size(thres_st,3)

t = [70 100];
% mtest = m_c{4,2,3};
% vartest = var_c{4,2,3};
vec = squeeze(sim_st(t(1),t(2),:) - m(t(1),t(2)));
vec = vec/sqrt(var(t(1),t(2)));
% vec = vec/sqrt(vartest(t(1),t(2)));
% vec = vec/std(vec)
param.visible = 'on';
param.save = 'on';
name = ['../article/figures/L2_p', num2str(p),'.png'];
compare_gaussian(vec,param,name)
%compare_gaussian(vec)

% sum(thres_test_st(t(1),t(2),:))/size(thres_st,3);
% mN(k) = abs(mean(vec));
% stdN(k) = abs(std(vec) - 1);
close all
pause(1)

%% ICDF VERSUS SIMILARITY

name = ['../article/figures/L2_cdf_p', num2str(p),'.png'];
[x,y,y_e] = compare_gaussian_cdf(vec,param,name);
export_curve(y(1:100:end), x(1:100:end),['../article/curves/L2_cdf_p', num2str(p), ...
                    '.tex']);
export_curve(y_e(1:100:end), x(1:100:end),['../article/curves/L2_cdf_p', num2str(p), '_empi.tex']);