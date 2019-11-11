function [sim_st] = internal_similarity_map(im_st,method,p)    

mask = zeros(size(im_st,1));
mask(1:p,1:p) = 1;

switch method
  case {'L2','L2asymp'}
    sim_st = sum(sum(im_st.^2 .* mask));
    sim_st = sim_st + ifft2(conj(fft2(mask)) .* fft2(im_st.^2));
    sim_st = sim_st - 2*ifft2(conj(fft2(mask .* im_st)) .* fft2(im_st));
   case {'ps','cos'}
    sim_st = ifft2(conj(fft2(mask .* im_st)) .* fft2(im_st));
    if strcmp(method, 'cos')
        factor = ifft2(conj(fft2(mask)) .* fft2(im_st.^2));
        factor = factor .* sum(sum(im_st.^2 .* mask));
        factor = sqrt(factor);
        sim_st = sim_st ./ factor;
    end
end
sim_st = real(sim_st);
        