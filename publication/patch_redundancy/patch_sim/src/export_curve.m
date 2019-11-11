function []=export_curve(x,y,filename)

fileID = fopen(filename, 'wt');
fprintf(fileID, '\\addplot coordinates { \n');
for k = 1:length(x)
    fprintf(fileID, ['(', num2str(x(k)) ,',', num2str(y(k)) ,') \n']);
end
fprintf(fileID, '};');
fclose(fileID);

