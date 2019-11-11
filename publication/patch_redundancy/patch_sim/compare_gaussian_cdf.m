function [x,y,y_empi] = compare_gaussian_cdf(vec,param,name)

if nargin==1
    param.visible = 'on';
    param.save = 'off';
end

[y_empi,x] = ecdf(vec);
m = min(vec(:));
M = max(vec(:));
prec = 10^-2;
y = normcdf(x);
if strcmp(param.visible,'off')
    figure('visible','off')
else
    figure;
end

figure;
plot(x,y,'LineWidth',3)
hold on
plot(x,y_empi,'r','LineWidth',3)
if strcmp(param.save,'on')
    export_fig(name, '-transparent')
end