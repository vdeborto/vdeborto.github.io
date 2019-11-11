function [valtab,tab] = best_neig(d,nb_neigh)
% function [valtab,tab] = best_neig(d,nb_neigh)
% This function takes a 2D matrix corresponding to a distance
% map and returns an array corresponding to the different
% positions of the nb_neigh lowest values.
% Input arguments:
%       -d: a 2D matrix corresponding to a distance map.
%       -nb_neigh: an integer, the number of lowest values
%       stored in the outputs.
% Output arguments:
%        -valtab: an array of size nb_neighx1 containing the
%        values of the nb_neigh lowest elements of d stored in
%        ascending order.
%        -tab: an array of size nb_neighx2 containing the
%        positions of the nb_neigh lowest elements of d stored
%        according the ascending order in valtab.
%       
% Valentin De Bortoli (01/11/2017)
[val,ind] = sort(d(:), 'ascend');
[tabx,taby] = ind2sub(size(d), ind(1:nb_neigh));
tab = [tabx taby];
valtab = val(1:nb_neigh);