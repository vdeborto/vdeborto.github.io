function [sim_st] = template_similarity_map(im_st,method,p,template)    

L = size(im_st,1);
mask = zeros(L,L);
mask(1:p,1:p) = 1;
L_template = size(template,1);
template_L = zeros(L,L,size(im_st,3));
if ndims(template) == 3
    template_st = template;
elseif ndims(template) == 2
    template_st = repmat(template, [1 1 size(im_st,3)]);
end
template_L(1:L_template,1:L_template,:) = template_st;

switch method
  case {'L2','L2asymp'}
    sim_st = sum(sum(template_L.^2 .* mask));
    sim_st = sim_st + ifft2(conj(fft2(mask)) .* fft2(im_st.^2));
    sim_st = sim_st - 2*ifft2(conj(fft2(mask .* template_L)) .* fft2(im_st));
  case {'ps','cos'}
    sim_st = ifft2(conj(fft2(template_L .* mask)) .* fft2(im_st));
    if strcmp(method, 'cos')
        sim_st = sim_st ./ sqrt(ifft2(conj(fft2(mask)) .* fft2(im_st.^2)) .* ...
                                      sum(sum(template_L.^2 .* mask)));
    end
end

sim_st = real(sim_st);