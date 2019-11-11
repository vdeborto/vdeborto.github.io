%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %                      ADSN test                       %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
clear res_st sres_st
n = 5000;
mode = 'Z';
L = 128;
im = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/img_18.png'])));
h_sz = [1 2 5 10 15 20 25];
p_sz = [5 10 15 20 30 40 50 70];
cellh{1} = '1';
cellh{2} = '2';
cellh{3} = '5';
cellh{4} = '10';
cellh{5} = '15';
cellh{6} = '20';
cellh{7} = '25';
cellv{1} = '5';
cellv{2} = '10';
cellv{3} = '15';
cellv{4} = '20';
cellv{5} = '30';
cellv{6} = '40';
cellv{7} = '50';
cellv{8} = '70';
method_c{1} = 'L2asymp';
method_c{2} = 'ps';
method_c{3} = 'cos';
nfa_t = [10];
N = length(h_sz) * length(p_sz) * length(method_c) * length(nfa_t);

for k = 1:N
    [k1,k2,k3,k4] = ind2sub([length(h_sz),length(p_sz),length(method_c),length(nfa_t)],k);
    h = im(1:h_sz(k1),1:h_sz(k1));
    im_st = adsn_generative(n,h,L,mode);
    p = p_sz(k2);
    if p<h_sz(k1)
        continue
    end
    template = im(1:p,1:p);
    method = method_c{k3};
    [m,var] = template_similarity_stat(h,method,p,L,template);
    sim_st = template_similarity_map(im_st,method,p,template);
    nfa = nfa_t(k4);
    [cdf_st,thres_st] = cdf_map(sim_st, m, var, p, method, nfa);
    %    vpv(dim_3Dstack(im_st), 'nc nw', dim_3Dstack(cdf_st), dim_3Dstack(cdf_st))
    thres_test_st = thres_st(1:L-p,1:L-p,:);
    %    sum(thres_test_st(:))/size(thres_st,3)
    %    sum(thres_test_st(t(1),t(2),:))/size(thres_st,3)
    res_c{k1,k2,k3,k4} = mean(thres_test_st,3);
    sres_c(k1,k2,k3,k4) = sum(sum(res_c{k1,k2,k3,k4}));
    m_c{k1,k2,k3,k4} = m;
    var_c{k1,k2,k3,k4} = var;
end
toc
