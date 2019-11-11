function [Iout] = write_patch_superp(I,p,tab,name)
%function [Iout] = write_patch_superp(I,p,tab,name)
% This function superposes patches of selected size on an
% original image.  Depending of if name argument variable is
% provided or not the output image is saved or not.
% Input arguments:
% 	-I: a 2D matrix (only grey images are
% 	supported), the original image.
% 	-p: an integer, the patch length.
% 	-tab: an array of size nbpatch x 2, containing
% 	the patches positions for the nbpatch patches to
% 	display.
% 	-name: (optional) a string. If an argument is
% 	provided then the superposed image is saved in
% 	your current directory under the name
% 	'name_superp.png'. All patches are also saved under
% 	the name 'name_patchnumber.png'
% Output arguments:
%        -Iout: a 3D matrix, the original grey image
%        superposed with the patches delimiters (red).
% See also: draw_patch, patch_extract, ind2str, new_range.
% Remark: if no name is provided then the function displays
% the output image using VideoProcessingViewer.
%
% Valentin De Bortoli (01/11/2017).
Iout = repmat(I, 1, 1, 3);
for k = 1:size(tab,1)
    Iout = draw_patch(Iout, tab(k,1), tab(k,2), p, 'draw', 'red');
    patch = patch_extract(I, tab(k,1), tab(k,2), p);
    if nargin==4
        imwrite(new_range(patch),[name, '_', ind2str(k,size(tab,1)), '.png'])
    end
end
if nargin<4
    vpv(Iout) %if vpv is not installed use imshow(new_range(Iout),[])
end
if nargin==4
    imwrite(new_range(Iout),[name, '_superp.png'])
end
