function [outminmap] = minmap(simmap)
% function [outminmap] = minmap(simmap)
% This function computes the local minima of a function. In 
% this context it used to find the local minima of a 
% similarity map (simmap). A local minimum in a discrete 
% setting is an ambiguous notion and here a 3x3 neighborhood
% is considered. In the ouput all values are put to Inf except
% the local minima which are left to their original values.
% Input arguments:
%       -simmap: a 2D matrix.
% Output arguments:
%        -outminmap: a 2D matrix of the same size of simmap.
%        It contains Inf values everywhere except at the local
%        minima where it has simmap values;
% See also: patch_extract
%
% Valentin De Bortoli (01/11/2017)
outminmap = Inf(size(simmap));
for t = 1:length(simmap(:))
    [x,y] = ind2sub(size(simmap),t);
    neighborhood = patch_extract(simmap,x,y,3,'center');
    if neighborhood(2,2)<=min(neighborhood(:))
        outminmap(x,y) = simmap(x,y);
    end
end
