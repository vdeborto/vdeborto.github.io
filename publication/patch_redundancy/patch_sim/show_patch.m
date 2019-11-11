function [Iout] = show_patch(I,p,tab,name)

%

Iout = repmat(I, 1, 1, 3);
for k = 1:size(tab,1)
    Iout = draw_patch(Iout, tab(k,1), tab(k,2), p, 'draw', 'red');
    patch = patch_extract(I, tab(k,1), tab(k,2), p);
    if nargin>=4
        imwrite(new_range(patch),[name, '_', ind2str(k,size(tab,1)), '.png'])
    end
end
%figure; imshow(Iout,[])
if nargin>=4
    imwrite(new_range(Iout),[name, '_superp.png'])
end
