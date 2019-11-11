function [] = compare_gaussian(vec,param,name)

if nargin==1
    param.visible = 'on';
    param.save = 'off';
end

m = min(vec(:));
M = max(vec(:));
prec = 10^-2;
x = m:prec:M;
y = normpdf(x);
if strcmp(param.visible,'off')
    figure('visible','off')
else
    figure;
end
histo = histogram(vec);
histo.Normalization = 'pdf';
hold on
plot(x,y,'LineWidth',3)
if strcmp(param.save,'on')
    export_fig(name, '-transparent')
end