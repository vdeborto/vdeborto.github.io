%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %           Perceptual similarity functions            %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('/home/debortoli/research/2017_internship/src/package/src/util/')

img_c{1} = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_07.png'])));
img_c{2} = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_18.png'])));
img_c{3} = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                   'img_11.png'])));
img_c{4} = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_16.png'])));
sigma_noise = 0;
for k = 1:4
    img_c{k} = img_c{k} + sigma_noise*randn(size(img_c{k}));
    img_c{k} = img_c{k} - mean(mean(img_c{k}));
end

img_name_c{1} = 'stochastic';
img_name_c{2} = 'periodic';
img_name_c{3} = 'contrast';
img_name_c{4} = 'meso';
L = size(img_c{1},1);
psim_c{1} = 'L2';
psim_c{2} = 'ps';
psim_c{3} = 'cos';
psim_c{4} = 'L1';
psim_c{5} = 'Linfinite';
%p_t = [5 7 10 15 20 40 60];
p_t = [20];

nb_neigh  = 20;
Nimg = length(img_c);
Npsim = length(psim_c);
Np = length(p_t);
Ntest = Nimg*Npsim*Np;

resmap = zeros(L,L,Ntest);
ressuperp = zeros(L,L,3,Ntest);
rest = zeros(1,Ntest);

param.matching_psim = 'internal';

for n = 1:Ntest
    [nimg,npsim,np] = ind2sub([Nimg, Npsim, Np], n);
    I = img_c{nimg};
    psim = psim_c{npsim};
    p = p_t(np);
    
    param.psize = p;
    param.func_psim = psim;    
    
    tic;sim_t = sim_function(I,param); time=toc;
    switch psim
      case 'L2'
        sim_t = sqrt(sim_t);
        sim_t = real(sim_t);
      case 'ps'
        sim_t = -sim_t;
      case 'cos'
        sim_t = -sim_t;
    end
    [f,x] = ecdf(sim_t(:));
    [valtab,tab] = best_neig(sim_t,nb_neigh);
    Isuperp = write_patch_superp(I,p,tab);
    Isuperp = draw_patch(Isuperp, 1, 1, p, 'draw', 'green', 2);
    %    keyboard
    resmap(:,:,n) = sim_t;
    ressuperp(:,:,:,n) = Isuperp;
    rest(n) = time;
    
    %save
    name = [img_name_c{nimg}, '_', ind2str(sigma_noise,50) '_', psim_c{npsim}, '_', ind2str(p,max(p_t(:))), ...
            '_'];
    preamble = '/home/debortoli/research/18_redundancy_gaussian/tex/article/';
    imwrite(new_range(sim_t), [preamble, name, 'simmap.png'])
    imwrite(new_range(Isuperp), [preamble, name, 'superp.png'])
    figure('visible','off');
    imagesc(sim_t)
    set(gca,'xtick',[],'ytick',[])
    colorbar
    %export_fig(['./res_percep/', name, 'superp_col'], '-eps', '-transparent')
    close all
    figure('visible','off');
    plot(x,f)
    %export_fig(['./res_percep/', name, 'ecdf'], '-eps', '-transparent')
end