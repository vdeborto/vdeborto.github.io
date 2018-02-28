%%%%% TP 2 OPTIMISATION NUMERIQUE %%%%%

clear all;
%close all;
figure

%% Exo 1 : Methode de Levenberg-Marquardt et Gauss-Newton.

x = [1:8]';
y = [0.127 0.2 0.3 0.25 0.32 0.5 0.7 0.9]';

epsilon = 10e-6;
mu = 1;

K = 2;
m = length(x);

beta = [1 1 1 1 3 6]';

f = @(x,beta) sum((ones(length(x),1)*beta(1:K)').*exp(-1/2*((x*ones(1,K)-(ones(length(x),1)*beta(2*K+1:3*K)')).^2)./(ones(length(x),1)*(beta(K+1:2*K)').^2)),2);

r = @(x,beta) y - f(x,beta);

expo = @(x,beta) [exp(-1/2*(x-beta(2*K+1)).^2/(beta(K+1)^2)) exp(-1/2*(x-beta(3*K)).^2/(beta(2*K)^2))];

%ex = expo(x,beta);

gradr = @(x,beta,ex) -[ ex(:,1) ex(:,2) beta(1)*(x-beta(2*K+1)).^2/(beta(K+1)^3).*ex(:,1) beta(K)*(x-beta(3*K)).^2/(beta(2*K)^3).*ex(:,2) beta(1)*(x-beta(2*K+1))/(beta(K+1)^2).*ex(:,1) beta(K)*(x-beta(3*K))/(beta(2*K)^2).*ex(:,2)]';

gradJ = 1;
%gradJ = @(x,beta,ex) gradr(x,beta,ex)*r(x,beta) + mu*beta;

i = 1;

while (norm(gradJ)>10^(-3)) && (i<1000)
    
    rk=(y-f(x,beta));%calcul du résidu
    gradrk=gradr(x,beta,expo(x,beta));%de son gradient
    A=gradrk*gradrk';%calcul de A pour en déduire muk
    %muk=10^(-4)*eigs(A,1);
    muk = epsilon*trace(A);
    
    gradJk=gradrk*rk+muk*beta;%calcul du gradient de J en betak (à la k-ième itération)
    Sk=gradrk*gradrk'+muk*eye(6);
    
    dk=-Sk\gradJk;%direction de descente choisie.
    rhok=0.02;
    beta=beta+rhok*dk;%on en déduit beta(k+1)
    i=i+1;
    
%     %ex = expo(x,beta);
%     aux = gradr(x,beta,expo(x,beta))*gradr(x,beta,expo(x,beta))';
%     %mu = epsilon*trace(aux);
%     mu = 10^-4*eigs(aux,1);
%     Sk =  gradr(x,beta,expo(x,beta))*gradr(x,beta,expo(x,beta))' + mu*eye(3*K);
%     
%     dk = -Sk\gradJ(x,beta,expo(x,beta));
%     rhok = 0.02;
%     beta = beta + rhok*dk;
%     i = i+1;
end

%     z1=[beta(1),beta(3),beta(5)];%alpha1,sigma1,x1
%     z2=[beta(2),beta(4),beta(6)];%alpha2,sigma2,x2
%     
%     rk=(Y-f(X,beta))';%calcul du résidu
%     gradrk=[-dalphaf(X,z1);-dalphaf(X,z2);-dsigmaf(X,z1);-dsigmaf(X,z2);-dxf(X,z1);-dxf(X,z2)];%de son gradient
%     A=gradrk*gradrk';%calcul de A pour en déduire muk
%     muk=10^(-4)*eigs(A,1);
%     
%     gradJk=gradrk*rk+muk*beta';%calcul du gradient de J en betak (à la k-ième itération)
%     Sk=gradrk*gradrk'+muk*eye(6);
%     
%     dk=-Sk\gradJk;%direction de descente choisie.
%     rhok=0.02;
%     beta=beta+rhok*dk';%on en déduit beta(k+1)
%     i=i+1;

plot(x,y,'*-k');
hold on
XX=[0:0.01:10]';
plot(XX,f(XX,beta),'r');
hold off


%% Exo 2 : Algorithmes sans gradient. Methode de Nelder et Mead

%% Exo 3 : Probleme de Lenard Jones de taille N