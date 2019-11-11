%% Illustration of patch repartition
addpath('/home/debortoli/research/2017_internship/src/package/src/util/')
save_dir = '/home/debortoli/research/18_redundancy_gaussian/tex/article/';

N = 64;
N_spot = 32;
pos1 = N/2 - N_spot/2;
pos2 = N/2 + N_spot/2;

f = zeros(N,N);
f(pos1:pos2,pos1:pos2) = 1;
f = f - mean(f(:));
f = f / std(f(:));
I = randn(N,N);
I_ADSN = randn(N,N);
I_ADSN = real(ifft2(fft2(f) .* fft2(I_ADSN)))/N;
%vpv('aw', 'ac', I, I_ADSN, f) %show original
%images + spot 
imwrite(new_range(I), [save_dir, 'white_noise_64.png'])
imwrite(new_range(I_ADSN), [save_dir, 'ADSN_64.png'])
imwrite(new_range(f), [save_dir, 'spot_64.png'])

%lifting in patch space 
N_p = 3;
vec_white_noise = zeros(N,N_p^2);
vec_ADSN = zeros(N,N_p^2);

for t=1:length(f(:))
    [x,y] = ind2sub([N N], t);
    I_shift = circshift(circshift(I, y-1, 2), x-1, 1);
    I_ADSN_shift = circshift(circshift(I_ADSN, y-1, 2), x-1, 1);
    vec_white_noise(t,:) = reshape(I_shift(1:N_p,1:N_p), 1, N_p^2) - ...
        reshape(I(1:N_p,1:N_p), 1, N_p^2);
    vec_ADSN(t,:) = reshape(I_ADSN_shift(1:N_p,1:N_p), 1, N_p^2) - ...
        reshape(I_ADSN(1:N_p,1:N_p), 1, N_p^2);
end

norm_white_noise = sqrt(sum(vec_white_noise.^2,2));
norm_ADSN = sqrt(sum(vec_ADSN.^2,2));
[~,ind] = sort(norm_white_noise);
[~,ind_ADSN] = sort(norm_ADSN);
N_pt = 20;
vec_white_noise_close = vec_white_noise(ind(1:N_pt),:);
vec_ADSN_close = vec_ADSN(ind_ADSN(1:N_pt),:);
[coeff,score,latent] = pca(vec_white_noise_close);
[coeff_ADSN,score_ADSN,latent_ADSN] = pca(vec_ADSN_close);

lift_white_noise = score(:,1:3);
lift_ADSN = score_ADSN(:,1:3);

stride = 1;
figure;
axis normal
scatter3(0,0,0, 50,'black','filled')
hold on
scatter3(score_ADSN(1:stride:end,1), score_ADSN(1:stride:end,2), score_ADSN(1: ...
                                                  stride:end,3), 20, 'r', ...
         'filled')
hold on
scatter3(score(1:stride:end,1), score(1:stride:end,2), score(1:stride:end,3), ...
         20, 'b', 'filled') 

X = [0.01 ; 0];
Y = X;
Z = X;
norm_ADSN_score =  max(sqrt(sum(score.^2,2)));
norm_white_noise_score =  max(sqrt(sum(score_ADSN.^2,2)));
hold on
scatter3sph(X,Y,Z,'size',[norm_white_noise_score, norm_ADSN_score],'color',[1 ...
                    0 0 ; 0 0 1],'transp',0.25)

export_fig([save_dir, '3dpoints.png'], '-transparent')

