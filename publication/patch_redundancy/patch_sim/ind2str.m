function [str] = ind2str(k,N)
% function [str] = ind2str(k,N)
% This function turns an indice into a string in order to
% write a filename. In order to know the number of digits
% used a maximal number for the integer must be provided.
% Input arguments:
%       -k: an integer, the indice to turn to string.
%       -N: an integer, the maximal number for the indice.
% Output arguments:
%        -str: a string, the indice seen as a string.
%
% Valentin De Bortoli (01/11/2017)
nbnum = floor(log10(N));
nbnumk = max(floor(log10(k+eps)),0);
diffz = nbnum - nbnumk;
str = [repmat('0',1,diffz), num2str(k)];