%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%% %% %  %                     LOCAL MINIMA                     %  % %% %%%%%
%%%%%%%%%% %% %% %  %                                      %  % %% %% %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

map = double(imread(['~/research/patch_sim/res_percep/' ...
                    'periodic_00_cos_20_simmap.png']));

psizet = [3 5 7 9 11 19];%must be odd
mapthrest = zeros(size(map,1),size(map,2),length(psizet));
for k = 1:length(psizet)
    psize = psizet(k);
    [id,mapthres] = local_minima(map,psize);
    mapthrest(:,:,k) = mapthres;
    %    vpv(mapthres)
end