function [id,mapthres] = local_minima(map,psize)

% p must be odd
mapthres = 0*map;
idx = [];
idy = [];
for k = 1:length(map(:))
    [x,y] = ind2sub(size(map), k);
    val = map(x,y);
    p = patch_extract(map,x,y,psize,'center');
    if all(p(:)>=val)
        idx = [idx x];
        idy = [idy y];
        mapthres(x,y) = 1;
    end
end
id = [idx ; idy];