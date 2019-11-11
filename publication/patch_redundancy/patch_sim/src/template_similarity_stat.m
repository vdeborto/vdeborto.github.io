function [m,var] = template_similarity_stat(h,method,p,L,template)

N_h = size(h,3);
L_h = size(h,1);
L_template = size(template,1);
h_L = zeros([L+L_h L+L_h N_h]);
template_L = h_L;
h_L(1:L_h,1:L_h,:) = h;
if ndims(template) == 3
    template_st = template;
elseif ndims(template) == 2
    template_st = repmat(template, [1 1 N_h]);
end
template_L(1:L_template,1:L_template,:) = template_st;
auto_h_L = real(ifft2(abs(fft2(h_L)).^2));

mask = zeros(L+L_h);
mask(1:p,1:p) = 1;
auto_mask = real(ifft2(abs(fft2(mask)).^2));

switch method
  case {'ps','cos'}
    m = zeros(1,1,N_h);
    var = sum(sum((mask.*template_L) .* ifft2(fft2(mask.*template_L).*fft2(auto_h_L))));
  if strcmp(method,'cos')
      factor = sqrt(p^2 * auto_h_L(1,1,:)) .* sqrt(sum(sum((mask.*template_L).^2)));
      m =  m./factor;
      var = var./factor^2;
  end
  case 'L2asymp'
    template_mask_L = mask.*template_L;
    m = p^2 * auto_h_L (1,1,:);
    m = m + sum(sum(template_mask_L.^2));
    var = ifft2(fft2(mask) .* fft2(auto_h_L.^2));
    var = var + 2 * template_mask_L .* ifft2(fft2(template_mask_L) .* ...
                                             fft2(auto_h_L));
    var = 2 * var;
    var = sum(sum(var));
end

if sum(var(:)<0)>0
    error('negative variance')
end

m = real(repmat(m, [L L]));
var = real(repmat(var, [L L]));
