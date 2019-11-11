function [m,var] = internal_similarity_stat(h,method,p,L)

L_h = size(h,1);
h_L = zeros([L+L_h L+L_h size(h,3)]);
h_L(1:L_h,1:L_h,:) = h;
auto_h_L = real(ifft2(abs(fft2(h_L)).^2));

mask = zeros(L+L_h);
mask(1:p,1:p) = 1;
auto_mask = real(ifft2(abs(fft2(mask)).^2));

crossed_term = db_conv(h,auto_mask,auto_h_L);

switch method
  case {'ps','cos'}
    m = p^2 * auto_h_L(1:L,1:L,:);
    var = sum(sum(auto_mask .* auto_h_L.^2));
    var = var + crossed_term;
%    var = 2 * sum(sum(auto_h_L.^2)) - auto_h_L(1,1).^2 - auto_h_L.^2 + 2 * crossed_term;
  if strcmp(method,'cos')
      factor = p^2 * auto_h_L(1,1,:);
      m =  m./factor;
      var = var./factor^2;
  end
  var = var(1:L,1:L,:);
  case 'L2asymp'
    m = 2 * p^2 * (auto_h_L(1,1) - auto_h_L(1:L,1:L,:));
    var = 4 * sum(sum(auto_mask .* auto_h_L.^2));
    var = var + 2 * real(ifft2(fft2(auto_mask) .* fft2(auto_h_L.^2)));
    var = var - 8 * real(ifft2(fft2(auto_mask .* auto_h_L) .* ...
                               fft2(auto_h_L)));
    var = var + 2 * crossed_term;
    var = 2 * var;
    var = var(1:L,1:L,:);
    var(1,1,:) = 0;
end

if sum(var(:)<0)>0
    error('negative variance')
end