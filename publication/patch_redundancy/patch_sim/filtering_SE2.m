N = 256;
N_square = 5;
N_square_theta = 5;
N_triangle = 10;
var = 1;
c = 8;

% defining square 
c = 8;
square = zeros(4*c-1);
square_N = zeros(N);
x = [-c/2 c/2 c/2 -c/2] + 2*c;
y = [-c/2 -c/2 c/2 c/2] + 2*c;
x_N = x - 2*c + N/2;
y_N = y - 2*c + N/2;
square = poly2mask(x,y,4*c-1,4*c-1);
square_N = poly2mask(x_N,y_N,N,N);
square_N = fftshift(square_N);

% defining rotated square
c = 8;
theta = 2 * pi * 30/360;
square_theta = zeros(4*c-1);
square_theta_N = zeros(N);
x = [-c/2 c/2 c/2 -c/2];
y = [-c/2 -c/2 c/2 c/2];
R_theta = [cos(theta) sin(theta) ;
          -sin(theta) cos(theta)];
Z = [x ; y];
Z_theta = R_theta * Z + 2*c;
x_theta = Z_theta(1,:);
y_theta = Z_theta(2,:);
x_theta_N = x_theta - 2*c + N/2;
y_theta_N = y_theta - 2*c + N/2;
square_theta = poly2mask(x_theta,y_theta,4*c-1,4*c-1);
square_theta_N = poly2mask(x_theta_N,y_theta_N,N,N);
square_theta_N = fftshift(square_theta_N);

% defining triangle
c = 8;
triangle = zeros(4*c-1);
triangle_N = zeros(N);
x = [2*c, 2*c-c/2, 2*c+c/2];
y = [2*c+floor(2/sqrt(3)*c)/2, 2*c-floor(2/sqrt(3)*c)/2, 2*c-floor(2/sqrt(3)* ...
                                                  c)/2];
x_N = x - 2*c + N/2;
y_N = y - 2*c + N/2;
triangle = poly2mask(x,y,4*c-1,4*c-1);
triangle_N = poly2mask(x_N,y_N,N,N);
triangle_N = fftshift(triangle_N);

%defining cross
L = 32;
l = 4;
m = N/2;
x = [m-l/2-L
     m-l/2-L
     m-l/2
     m-l/2
     m+l/2
     m+l/2
     m+l/2+L
     m+l/2+L
     m+l/2
     m+l/2
     m-l/2
     m-l/2]';
y = circshift(x,-3);
cross_N = poly2mask(x,y,N,N);
cross_N = fftshift(cross_N);

epsilon = 1
N_test = 10;
N_std = 20;
res = zeros(N_test,N_std,3);
k_sim = 0;
    f_psim_l = {'L2','L1','Linfinite'};
for k_sim=1:3
f_psim = f_psim_l{k_sim};
for n_test = 1:N_test

% filling image
u = zeros(N);
pos_square = randi(N, 2, N_square);
pos_square_theta = randi(N, 2, N_square_theta);
pos_triangle = randi(N, 2, N_triangle);

u_square = u;
u_square_theta = u;
u_triangle = u;

for i = 1:N_square
    u_square(pos_square(1,i), pos_square(2,i)) = 1;
end

for i = 1:N_square_theta
    u_square_theta(pos_square_theta(1,i), pos_square_theta(2,i)) = 1;
end

for i = 1:N_triangle
    u_triangle(pos_triangle(1,i), pos_triangle(2,i)) = 1;
end

square = u * 0;
square(1:c,1:c) = 1; %juste pour l'analyse des positions si on veut une
                       %même pipeline pour tous les motifs il faut remplacer
                       %square par square_N
u = real(ifft2(fft2(u_square) .* fft2(square))) + real(ifft2(fft2(u_triangle) ...
                                                  .* fft2(triangle_N)));
u = u + real(ifft2(fft2(u_square_theta) .* fft2(square_theta_N)));
u = 1.*(u>0.5);
fprintf(['iteration number ', num2str(n_test) '\n'])
%vpv(u)

%% filtering

% eps_thres = 0.01;
% square_corr = real(ifft2(fft2(u) .* fft2(square_N)));
% square_theta_corr = real(ifft2(fft2(u) .* fft2(square_theta_N)));
% triangle_corr = real(ifft2(fft2(u)) .* fft2(triangle_N));
% thresh_sq = max(square_corr(:)) - eps_thres * range(square_corr(:));
% thresh_tr = max(triangle_corr(:)) - eps_thres * range(triangle_corr(:));
% pos_square_guess = 1.*(square_corr > thresh_sq);
% pos_square_theta_guess = 1.*(square_theta_corr > thresh_sq);
% pos_triangle_guess = 1.*(triangle_corr > thresh_tr);

% % displaying

% u_col = repmat(u,1,1,3);
% u_col_rot = repmat(u,1,1,3);
% cross_sq = real(ifft2(fft2(pos_square_guess) .* fft2(cross_N))) > 0.9;
% cross_sq_rot = real(ifft2(fft2(pos_square_guess+pos_square_theta_guess) .* fft2(cross_N))) > 0.9;
% u_col(:,:,1) = (squeeze(u_col(:,:,1)) + cross_sq) > 0.9;
% u_col(:,:,2) = (squeeze(u_col(:,:,2)) - cross_sq) > 0.9;
% u_col(:,:,3) = (squeeze(u_col(:,:,3)) - cross_sq) > 0.9;
% u_col_rot(:,:,1) = (squeeze(u_col_rot(:,:,1)) + cross_sq_rot) > 0.9;
% u_col_rot(:,:,2) = (squeeze(u_col_rot(:,:,2)) - cross_sq_rot) > 0.9;
% u_col_rot(:,:,3) = (squeeze(u_col_rot(:,:,3)) - cross_sq_rot) > 0.9;

% % saving
% filename = ['square_', num2str(N_square) '_triangle_', num2str(N_triangle) ...
%             '_squaretheta_', num2str(N_square_theta) '.png'];
% imwrite(u, ['/home/debortoli/research/harmonic/resources/misc/original_', filename])
% imwrite(u_col, ['/home/debortoli/research/harmonic/resources/misc/result_', ...
%                 filename])
% imwrite(u_col_rot,
% ['/home/debortoli/research/harmonic/resources/misc/result_rot_', filename])

k = 0;

for std=linspace(0,0.5,N_std)
    k =  k +1;
    param_sim.func_psim = f_psim;
    param_sim.matching_psim = 'template';
    param_sim.template = ones(3);
    sim = sim_function(u+std*randn(size(u)), param_sim);
    sim_vec = sim(:);
    res(n_test,k,k_sim) = sum(sim_vec<epsilon);
    %    [sort_sim_vec, idx] = sort(sim_vec);
    %    [valuuu, idx_sq] = sort(u_square(:), 'descend');
    %    res(n_test,k) = sum(abs(sort(idx(1:N_square)) - sort(idx_sq(1: ...
    %                                                      N_square))))/(N_square ...
    %                                                      * N^2);
    end                                     
end
end
res_m = squeeze(mean(res,1))

figure;
for k_sim=1:3
    hold on
    plot(1:length(res_m(:,k_sim)), res_m(:,k_sim))
end
