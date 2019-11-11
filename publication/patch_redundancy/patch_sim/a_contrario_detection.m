function [d_true, eig_m, prob_map] = a_contrario_detection(input, param_detect)
%[d_true, eig_m, prob_map] = a_contrario_detection(input, param_detect)
%
%This function computes a detection of offsets based on a contrario model
%built with a ADSN noise model.
%
%Inputs:
%   -input: original image of size Lx x Ly or a string, the adress of the
%   original image of size Lx x Ly. input can also be a distance map if 
%   param_detect.input = 'map'.
%   -param_detect: (optional, see remark) a structure containing all the 
%   parameters of the algorithm:
%       -px: x-position (up-left) of the patch in I.
%       -py: y-position (up-left) of the patch in I.
%       -Lp: patch height (all considered patches are squared).
%       -thres: probability threshold in the detection model. NFA = thres x
%       number of pixels in the image.
%       -spot_option: a string. If spot_option = 'ADSN' then the noise 
%       model used is the ADSN model. If spot_option = 'white_noise' then 
%       the noise model used is the white noise mode.
%       -eig_option: a string. If eig_option = 'true' then we compute the
%       eigenvalues of the covariance matrix using Matlab routines. If
%       eig_option = 'approx_frob' then we compute an approximate version 
%       of the eigenvalues using the projection on circulant matrices. See
%       approx_frob for more details.
%       -inv_option: a string. If inv_option = 'direct' then the model is
%       computed using the cdf. If inv_option = 'inverse' then the model is
%       computed using the icdf. We recommend using the 'direct'
%       implementation.
%       -cdf_option: a string. If cdf_option = 'imhof' then the cdf (or 
%       icdf depending on inv_option) of a weighted sum of chi-square 
%       variables is computed using Imhof approximation. If cdf_option = 
%       'hbe' then this same cdf (or icdf) is computed using the
%       Hall-Buckley-Eagleson approximation (faster but less precise). If
%       cdf_option = 'Wood F' then the Wood F method is chosen. One last
%       option is cdf_option = 'normal' which performs the normal
%       approximation (should not be used in practice). We recommend using
%       Wood F approximation.
%       -fill_method: a string. See approx_frob_loop for more details about
%       this option. If this field is void then 'offset_patch' is chosen
%       the other option being 'offset_image'.
%       -input: a string. Default is 'image'. If input = 'image' then input
%       must contain an image (see input). If input = 'map' then input is a
%       matrix. In this case param_detect.spot must be non void.
%       -spot: in the case where param_detect.input = 'map', 
%       param_detect.spot is a matrix the same size of input and contains 
%       the original spot.    
%
%Outputs:
%   -d_true: a binary image. 1 means the associated position is detected.
%   In order to turn this position map into an offset map one must operate
%   a circular (x,y) shift on d_true.
%   -eig_m: a matrix of size Lx x Ly x Lp^2, where eig_m(x,y,:) is the
%   set of all the eigenvalues for the offset (x,y).
%   -prob_map: a map containing all the probability computed in the direct
%   mode. d_map_true = prob_map<thres.
%
%Remark: if no param_detect structure is entered then default is:
%   -param_detect.px = 1;
%   -param_detect.py = 1;
%   -param_detect.Lp = 15;
%   -param_detect.thres = 10^(-6);
%   -param_detect.spot_option = 'ADSN';
%   -param_detect.eig_option = 'approx_frob';
%   -param_detect.inv_option = 'direct';
%   -param_detect.cdf_option = 'woodf';
%   -param_detect.fill_method = 'offset_patch';
    if nargin<2 %no parameter structure is entered
        param_detect = struct;
        param_detect.input = 'image';
        param_detect.px = 1;
        param_detect.py = 1;
        param_detect.Lp = 15;
        param_detect.thres = 10^(-6);
        param_detect.spot_option = 'ADSN';
        param_detect.eig_option = 'approx_frob';
        param_detect.inv_option = 'direct';
        param_detect.cdf_option = 'woodf';
        param_detect.fill_method = 'offset_patch';
    end
    
    if strcmp(param_detect.input,'image')
        if ischar(input) == 1
            I = imread(input);
        else
            I = input;
        end
        [Lx,Ly,~] = size(I);
    elseif strcmp(param_detect.input,'map')
        d_map_true = input;
        [Lx,Ly,~] = size(input);
    end

    px = param_detect.px;
    py = param_detect.py;
    Lp = param_detect.Lp;
    thres = param_detect.thres;
    spot_option = param_detect.spot_option;
    eig_option = param_detect.eig_option;
    cdf_option = param_detect.cdf_option;
    inv_option = param_detect.inv_option;
    fill_method = param_detect.fill_method;
    
    if strcmp(param_detect.input,'image')
        d_map_true = zeros(Lx,Ly);
    end
    prob_map = zeros(Lx,Ly);
    d_true = zeros(Lx,Ly);

    %% autocorrelation of original image
    
    if strcmp(param_detect.input,'image')
    if ndims(I) == 3
        I = double(rgb2gray(I));
    else
        I = double(I);
    end

    I = I - mean(I(:));

    switch spot_option
      case 'ADSN'  
        auto_spot = ifft2(abs(fft2(I)).^2) / (Lx*Ly); 
      case 'white_noise'
        auto_spot = zeros(Lx,Ly);
        auto_spot(1,1) = 1/(Lx*Ly - 1)*sum(sum(I.^2));
    end
    elseif strcmp(param_detect.input,'map')
        spot = param_detect.spot;
        auto_spot = ifft2(abs(fft2(spot)).^2)/ (Lx*Ly);
    end

    %% patch distance map
    
    if strcmp(param_detect.input,'image')
        d_map_true = distance_patch_shift(I, px, py, Lp); %build distance map
    end

    %% ADSN patch distance map

    switch eig_option
      case 'true'
        mat = func2mat(auto_spot, Lp); %build first part of the offset correlation matrix
        mat = 2*mat;
        eig_m = zeros(Lp^2, Lx*Ly);
        
        parfor t = 1:Lx*Ly
            [tx,ty] = ind2sub([Lx Ly], t);
            ttx = tx - 1;
            tty = ty - 1;
            auto_spot_t = circshift(circshift(auto_spot, ttx, 1), tty, 2);
            mat_t = func2mat(auto_spot_t, Lp); %build offset dependent part of the offset correlation matrix
                                               %we do not use offset_correlation_function so the offset
                                               %independent part of the offset correlation matrix can be
                                               %computed outside of the (par)for loop.
            M = mat - mat_t - mat_t';
            e = eig(M); %find eigenvalues of the offset correlation matrix
            eig_m(:,t) = e;
        end

      case 'approx_frob'
        eig_m = approx_frob_loop(auto_spot, Lp, fill_method); %compute approximate eigenvalues for every offset
    end

    %% detection

    switch inv_option
      case 'direct'
        prob_map = cdf_wchi(eig_m, 'direct', d_map_true(:)', ...
                            cdf_option);
        prob_map(1) = 0;
        prob_map = reshape(prob_map, Lx, Ly);
        d_true = prob_map < thres;
      case 'inverse'
        eig_m(:,1) = ones(Lp^2,1); %no infinite loop in Imhof
        d_thres = cdf_wchi(eig_m, 'inverse', thres, cdf_option, thres/10);
        d_thres(1) = 0;
        d_thres = reshape(d_thres, Lx, Ly);
        d_true = d_map_true < d_thres;
    end

    %% centering of the detection figure

    if strcmp(inv_option, 'inverse')
        d_true = circshift(circshift(d_true, px-1, 1), py-1, 2);
    else 
        d_true = circshift(circshift(d_true, px-1, 1), py-1, 2);
        prob_map = circshift(circshift(prob_map, px-1, 1), py-1, 2);
    end

    eig_m = reshape(eig_m, Lp^2, Lx, Ly);
    eig_m = circshift(circshift(eig_m, px-1, 2), py-1, 3);
    eig_m = sort(real(eig_m), 1, 'descend'); %reshape eig_m in a more readable way
end
