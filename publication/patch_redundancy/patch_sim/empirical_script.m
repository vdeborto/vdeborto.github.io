%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %         Empirical patch similarity functions         %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imtest = double(rgb2gray(imread(['~/img/simoncelli/simoncelli_original/' ...
                    'img_08.png'])));
imtest = imtest - mean(imtest(:));
N = 1000;
L = 256;
p = 60;
t = [10,20];
h = zeros(L);
spotsize = 10;
spot = imtest(1:spotsize,1:spotsize);
facnorm = sqrt(sum(imtest(:).^2)/sum(spot(:).^2));
spot = facnorm * spot; %normalizing the spot = same variance as the whole image
%spot = ones(1);
%figure; imshow(spot,[])
hshow = h + min(min(spot(:)),0);
hshow(1:spotsize,1:spotsize) = spot;
hshow = hshow - min(hshow(:));
h(1:spotsize,1:spotsize) = spot;
name = 'fullADSN';
%h = h-mean(h(:));
vpv(hshow)
imwrite(new_range(fftshift(circshift(circshift(hshow,-p/2,1),-p/2,2))), ['./res/', name, '.png'])
Xdata = real(ifft2(fft2(randn([L L N])) .* fft2(h))/L);
imwrite(new_range(squeeze(Xdata(:,:,1))), ['./res/', name, '_sample.png'])
fprintf(['maximum imaginary part (absolute value): ', ...
         num2str(max(abs(imag(Xdata(:))))), '\n'])
template = rand(p);
template = template/sqrt(sum(sum(template.^2)));
func_psim = cell(1,5);
matching_psim = cell(1,2);
func_psim{1} = 'L2';
func_psim{2} = 'ps';
func_psim{3} = 'cosine';
func_psim{4} = 'Linfinite';
func_psim{5} = 'L1';
func_psim{6} = 'L2asymp';
matching_psim{1} = 'internal';
matching_psim{2} = 'template';

param.name = name;
param.save = 'on';
param.visible = 'on';
param.data = Xdata;
param.func_psim_c = func_psim;
param.matching_psim_c = matching_psim;
param.template = template;
param.psize = p;
param.poffset = t;
param.spot = h;
param.spotsize = spotsize;

tic;
[empirical_psim_m] = empirical_psim(param);
time = toc;
fprintf(['time elapsed computing empirical values: ', num2str(time), 's\n'])

empirical_comp(empirical_psim_m,param)