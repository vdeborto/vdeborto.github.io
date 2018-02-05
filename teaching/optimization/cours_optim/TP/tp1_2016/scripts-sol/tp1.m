%%% TP1 2014-2015 - Methode de gradient, gradient conjugu� et quasi-Newton

clear all
close all

J=@(u1,u2) (u1-1).^2 + 100*(u1.^2 - u2).^2;%fonction de Rosenbrock
gradJ = @(u1,u2) [2*(u1-1)+4*100*u1*(u1^2-u2), -2*100*(u1^2-u2)];

%% exo 1 - Fonction de Rosenbrock et gradient a pas fixe

% [u1,u2] = meshgrid(-1.5:0.01:1.5, -0.5:0.01:1.5);
% 
% Z = J(u1,u2);
% figure(1); clf
% mesh(u1,u2,Z)
% 
% figure(2); clf
% contourf(u1,u2,Z,50);
% 
% u0 = [-1,1];
% rho = 0.002;%2*10^-3, 0.0045, 0.01
tol = 10^-4;
% max_it = 5000;
% 
% u = u0;
% 
% hold on
% plot(u(1),u(2),'*r')
% plot(1,1,'*g')
% it = 0;
% while (J(u(1),u(2))>=tol && it<max_it)
%     gradJ(u(1),u(2))/norm(gradJ(u(1),u(2)))
%     u = u - rho*gradJ(u(1),u(2));
%     %u = u - rho*gradJ(u(1),u(2))/norm(gradJ(u(1),u(2)));
%     plot(u(1),u(2),'-r')
%     %pause(0.2)
%     it = it+1;
%     title(['EXO 1 : it = ',num2str(it)])
%     drawnow
% end
% plot(u(1),u(2),'r*')
% hold off

%% exo 2 - Recherche lin�aire - Regle d'Aramijo

% L = 100;
% m = 0.4;
% beta = 0.5;
% 
% max_it = 10000; %nb d't�ration max
% 
% u0 = [-1 1];
% 
% u = u0;
% 
% %plot J
% [U1, U2] = meshgrid(-1.5:.05:1.5,-0.5:.05:1.5);
% v1 = U1(:);
% v2 = U2(:);
% figure(3)
% clf
% %mesh(U1, U2, reshape( J(U1, U2, e),size(U1,1), size(U1,2)) )
% %mesh(U1, U2, reshape( J([v1'; v2']),size(U1,1), size(U1,2)) )
% contourf(U1, U2, reshape( J(v1, v2),size(U1,1), size(U1,2)) )
% 
% hold on
% 
% %trajectoire
% plot(u(1),u(2),'r*')
% plot(1,1,'m*')
% 
% %calcul de alphak
% for k = 1:max_it
%     
%     %compute new alpha
%     alpha = 1/L;
%     gJ = gradJ(u(1),u(2));
%     dk = -gJ;
%     normGJ2 = gJ*dk';
%     
%     while( J(u(1) + alpha*dk(1),u(2) + alpha*dk(2)) > J(u(1),u(2)) + m*alpha*normGJ2 )
%         alpha = alpha*beta;
%     end
%     
%     u = u + alpha*dk;
%     
%     plot(u(1),u(2),'--c*','LineWidth',1)
%     pause(0.2)
%     title(['Aramijo - it= ', num2str(k)])
%     drawnow
%                 
% end
% 
% hold off


%% exo 3 - Gradient conjugu�

L = 100;
m = 0.4;
B = 0.5;
itmax = 100;

[u1,u2] = meshgrid(-1.5:0.01:1.5, -0.5:0.01:1.5);

Z = J(u1,u2);
figure;
contourf(u1,u2,Z,20);
hold on

u = [-1,1];
plot(u(1),u(2),'*r')
d = -gradJ(u(1),u(2));
gradOld = gradJ(u(1),u(2));

it = 1;
while (J(u(1),u(2))>=tol)

    alpha = - (gradOld*d')/(L*(d*d'));
    count = 0;
    while(J(u(1)+alpha*d(1),u(2)+alpha*d(2)) > J(u(1),u(2)) + m*alpha*(gradOld*d'))
        count = count+1;
        alpha = alpha*B;
    end
    
    alpha;
    u = u+alpha*d;
    plot(u(1),u(2),'*r')
    title(['it= ', num2str(it)])
    drawnow
    
    beta = (gradJ(u(1),u(2))-gradOld)*(gradJ(u(1),u(2))')/(gradOld*gradOld')
    %beta = (gradJ(u(1),u(2))-gradOld)*(gradJ(u(1),u(2))')/(d*(gradJ(u(1),u(2))-gradOld)');
    d = -gradJ(u(1),u(2)) + beta*d;
    gradOld = gradJ(u(1),u(2));
    
    it = it+1;
    J(u(1),u(2))
    
end


%% exo 4 - Lenard-Jones de taille N

% V = @(r) 1./(r.^12) - 2./(r.^6);
% gV = @(r) -12./(r.^13) + 12./(r.^7);
% 
% N = 4;
% L = 100;
% m = 0.4;
% beta = 0.5;
% 
% X1 = [0 0 0];
% X2 = rand(1,3)*2*(4^(1/3)) - 4^(1/3);
% X3 = rand(1,3)*2*(4^(1/3)) - 4^(1/3);
% X4 = rand(1,3)*2*(4^(1/3)) - 4^(1/3);
% 
% u0 = [X1',X2', X3', X4'];
% LJN(u0,N,V)
% 
% % figure
% % plot(1:0.1:10,V(1:0.1:10))
% 
% figure
% plot3(X1(1), X1(2), X1(3), 'm*', 'LineWidth', 3)
% title('EXO 5 : Lenard Jones')
% hold on
% plot3(X2(1), X2(2), X2(3), 'r*', 'LineWidth', 3)
% plot3(X3(1), X3(2), X3(3), 'b*', 'LineWidth', 3)
% plot3(X4(1), X4(2), X4(3), 'k*', 'LineWidth', 3)
% grid on
% xlabel('x')
% ylabel('y')
% zlabel('z')
% 
% 
% 
% d0 = -LJN(u0,N,gV);
% 
% alpha = 1/L;
% gJ = LJN(u0,N,gV);
% dk = d0;
% normGJ2 = gJ*dk';
% 
% while( J(u0(1) + alpha*dk(1),u0(2) + alpha*dk(2)) > J(u0(1),u0(2)) + m*alpha*normGJ2 )
%     alpha = alpha*beta;
% end
% 
% u1 = u0 + alpha*d0;
% 
% 
% u2  = u1;
% u1 = u0; 
% u = u2;
% d = d0;
% 
% 
% 
% while(it <= it_max && diam>1e-5)
%     
%     
% it = 1;
% while (it <= itmax)
%     gJ1 = gradJ(u1(1),u1(2));
%     gJ2 = gradJ(u2(1),u2(2));
%     beta = -((gJ2-gJ1)*gJ1')/(gJ1*gJ1');
%     d = -gradJ(u1(1),u1(2)) + beta*d
%     
%     %compute new alpha
%     alpha = 1/L;
%     gJ = gradJ(u2(1),u2(2));
%     dk = d;
%     normGJ2 = gJ*dk';
%     
%     while( J(u2(1) + alpha*dk(1),u2(2) + alpha*dk(2)) > J(u2(1),u2(2)) + m*alpha*normGJ2 )
%         alpha = alpha*beta;
%     end
%     
%     u = u2 + alpha*d
%     
%     plot(u(1),u(2),'--c*','LineWidth',1)
%     title(['It= ', num2str(it)])
%     drawnow
%     
%     it = it+1;
% 
% end    
%     
%     
% end
% 
% hold off
% 
% XX = [0, x(1,1), x(2,1), x(4,1); 0, 0, x(3,1), x(5,1); 0, 0, 0, x(6,1)];
% figure
% plot3(XX(1,:),XX(2,:),XX(3,:),'*r','LineWidth',2)
% xlabel('x')
% ylabel('y')
% zlabel('z')
% grid on
% hold on
% plot3([-2 2], [0 0], [0 0], 'b', 'LineWidth', 1.5)
% plot3([0 0], [-2 2], [0 0], 'b', 'LineWidth', 1.5)
% plot3([0 0], [0 0], [-2 2],  'b', 'LineWidth', 1.5)
 %axis([-2,2,-2,2,-2,2])


% %% exo 5 - M�thode de Levenberg-Marquadt et Gauss Newton . Application a la
% %  regression non-lin�aire
% 
% x = [1:8]';
% y = [0.127 0.2 0.3 0.25 0.32 0.5 0.7 0.9]';
% 
% epsilon = 10e-6;
% mu = 1;
% 
% K = 2;
% m = length(x);
% 
% beta = [1 1 1 1 3 6]';
% 
% f = @(x,beta) sum((ones(length(x),1)*beta(1:K)').*exp(-1/2*((x*ones(1,K)-(ones(length(x),1)*beta(2*K+1:3*K)')).^2)./(ones(length(x),1)*(beta(K+1:2*K)').^2)),2);
% 
% r = @(x,beta) y - f(x,beta);
% 
% expo = @(x,beta) [exp(-1/2*(x-beta(2*K+1)).^2/(beta(K+1)^2)) exp(-1/2*(x-beta(3*K)).^2/(beta(2*K)^2))];
% 
% %ex = expo(x,beta);
% 
% gradr = @(x,beta,ex) -[ ex(:,1) ex(:,2) beta(1)*(x-beta(2*K+1)).^2/(beta(K+1)^3).*ex(:,1) beta(K)*(x-beta(3*K)).^2/(beta(2*K)^3).*ex(:,2) beta(1)*(x-beta(2*K+1))/(beta(K+1)^2).*ex(:,1) beta(K)*(x-beta(3*K))/(beta(2*K)^2).*ex(:,2)]';
% 
% gradJ = 1;
% %gradJ = @(x,beta,ex) gradr(x,beta,ex)*r(x,beta) + mu*beta;
% 
% i = 1;
% 
% while (norm(gradJ)>10^(-3)) && (i<1000)
%     
%     rk=(y-f(x,beta));%calcul du résidu
%     gradrk=gradr(x,beta,expo(x,beta));%de son gradient
%     A=gradrk*gradrk';%calcul de A pour en déduire muk
%     %muk=10^(-4)*eigs(A,1);
%     muk = epsilon*trace(A);
%     
%     gradJk=gradrk*rk+muk*beta;%calcul du gradient de J en betak (à la k-ième itération)
%     Sk=gradrk*gradrk'+muk*eye(6);
%     
%     dk=-Sk\gradJk;%direction de descente choisie.
%     rhok=0.02;
%     beta=beta+rhok*dk;%on en déduit beta(k+1)
%     i=i+1;
%     
% %     %ex = ex)po(x,beta);
% %     aux = gradr(x,beta,expo(x,beta))*gradr(x,beta,expo(x,beta))';
% %     %mu = epsilon*trace(aux);
% %     mu = 10^-4*eigs(aux,1);
% %     Sk =  gradr(x,beta,expo(x,beta))*gradr(x,beta,expo(x,beta))' + mu*eye(3*K);
% %     
% %     dk = -Sk\gradJ(x,beta,expo(x,beta));
% %     rhok = 0.02;
% %     beta = beta + rhok*dk;
% %     i = i+1;
% end
% 
% %     z1=[beta(1),beta(3),beta(5)];%alpha1,sigma1,x1
% %     z2=[beta(2),beta(4),beta(6)];%alpha2,sigma2,x2
% %     
% %     rk=(Y-f(X,beta))';%calcul du résidu
% %     gradrk=[-dalphaf(X,z1);-dalphaf(X,z2);-dsigmaf(X,z1);-dsigmaf(X,z2);-dxf(X,z1);-dxf(X,z2)];%de son gradient
% %     A=gradrk*gradrk';%calcul de A pour en déduire muk
% %     muk=10^(-4)*eigs(A,1);
% %     
% %     gradJk=gradrk*rk+muk*beta';%calcul du gradient de J en betak (à la k-ième itération)
% %     Sk=gradrk*gradrk'+muk*eye(6);
% %     
% %     dk=-Sk\gradJk;%direction de descente choisie.
% %     rhok=0.02;
% %     beta=beta+rhok*dk';%on en déduit beta(k+1)
% %     i=i+1;
% 
% plot(x,y,'*-k');
% hold on
% XX=[0:0.01:10]';
% plot(XX,f(XX,beta),'r');
% hold off
