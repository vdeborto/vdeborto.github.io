%TP1
clear all; close all

%% Exercice 1 - fonction de Rosenbrock et gradient a pas fixe
% 
J = @(u1,u2,e) (u1-1).^2 + e*(u1.^2 - u2).^2;

e = 100;
[u1,u2] = meshgrid(-1.5:0.05:1.5, -0.5:0.05:1.5);

Z = J(u1,u2,e);
%mesh(u1,u2,Z)

figure(1);
clf
contourf(u1,u2,Z,20);

u0 = [-1,0];
rho = 0.01;
tol = 10^-3;
max_it = 500;%10000;

gradJ = @(u1,u2,e) [2*(u1-1)+4*e*u1*(u1^2-u2), -2*e*(u1^2-u2)];

u = u0;

hold on
plot(u(1),u(2),'*r')
it = 0;
while (J(u(1),u(2),e)>=tol && it<max_it)
    %u = u - rho*gradJ(u(1),u(2),e)
    u = u - rho*gradJ(u(1),u(2),e)/norm(gradJ(u(1),u(2),e));
    plot(u(1),u(2),'r*')
    %pause(0.2)
    it = it+1;
    title(['EXO 1 : it = ',num2str(it)])
    drawnow
end
hold off
it


%% Exercice 2 POSITION D'EQUILIBRE D'UN FIL ELASTIQUE SUSPENDU

n = 8;
A = [0 5];
B = [10 5];
m = 0.1; 
K = 10;
g = 9.81;

J = @(X,Y,K,m,g) 1/2*K*ones(1,n+1)*( (X(2:n+2)-X(1:n+1)).^2 + (Y(2:n+2)-Y(1:n+1)).^2 ) + m*g*ones(1,n)*Y(2:n+1);

gradJ_x = @(x,K) K*[(2*x(2) - x(1) - x(3)) (2*x(3) - x(2) - x(4)) (2*x(4) - x(3) - x(5)) (2*x(5) - x(4) - x(6)) (2*x(6) - x(5) - x(7)) (2*x(7) - x(6) - x(8)) (2*x(8) - x(7) - x(9)) (2*x(9) - x(8) - x(10))]';
gradJ_y = @(y,K,m,g) K*[(2*y(2) - y(1) - y(3)) (2*y(3) - y(2) - y(4)) (2*y(4) - y(3) - y(5)) (2*y(5) - y(4) - y(6)) (2*y(6) - y(5) - y(7)) (2*y(7) - y(6) - y(8)) (2*y(8) - y(7) - y(9)) (2*y(9) - y(8) - y(10))]' + m*g*ones(n,1);

deltaX = (B(1)-A(1))/(n+1);
deltaY = (B(2)-A(2))/(n+1);

X0 = A(1)*ones(n,1) + deltaX*[1:n]';
Y0 = A(2)*ones(n,1) + deltaY*[1:n]';

rho = 0.001;
tol = 10^-3;
max_it = 1000;%10000;

X = X0;
Y = Y0;

it = 0;
figure(2)
clf
while (J([A(1); X; B(1)],[A(2); Y; B(2)],K,m,g)>=tol && it<max_it)
    X = X - rho*gradJ_x([A(1); X; B(1)],K);
    Y = Y - rho*gradJ_y([A(2); Y; B(2)],K,m,g);
    plot([A(1);X;B(1)],[A(2);Y;B(2)])
    hold on
    axis([0 10 3 5])
    plot([A(1);X;B(1)],[A(2);Y;B(2)],'r*')
    hold off
    it = it+1;
    title(['EXO 2 : Position d equilibre du fil elastique it=', num2str(it)])
    drawnow
end
it

%% Exercice 3 : Règle d'Aramijo

J = @(u) (u(1,:)-1).^2 + 100*(u(1,:).^2 - u(2,:)).^2; % fonction rosenbrock
GJ = @(u) [2*(u(1,:)-1) + 2*(u(1,:).^2 - u(2,:)).*2.*u(1,:); -100*2*(u(1,:).^2 - u(2,:))];
dq0 = @(u,gj) [ (2*(u(1,:)-1).*(-gj(1,:)) + 100*2*(u(1,:).^2-u(2,:)).*(2*u(1,:).*(-gj(1,:))+gj(2,:))).*(-gj(1,:)); (2*(u(1,:)-1).*(-gj(1,:)) + 100*2*(u(1,:).^2-u(2,:)).*(2*u(1,:).*(-gj(1,:))+gj(2,:))).*(-gj(2,:))];

rhok = @(R,k) R/(1 + (k/10));

R = 3;
max_it = 300; %nb d'tération max
m = 0.3;

u0 = [-1; 0];

u = u0;

%plot J
[U1, U2] = meshgrid(-1.5:.05:1.5,-0.5:.05:1.5);
v1 = U1(:);
v2 = U2(:);
figure(3)
clf
%mesh(U1, U2, reshape( J(U1, U2, e),size(U1,1), size(U1,2)) )
%mesh(U1, U2, reshape( J([v1'; v2']),size(U1,1), size(U1,2)) )
contourf(U1, U2, reshape( J([v1'; v2']),size(U1,1), size(U1,2)) )

hold on

%trajectoire
plot(u(1),u(2),'r*')

%calcul de alphak
for k = 1:max_it
    
    %compute new alpha
    alpha = rhok(R,k);
    
    gradJ = GJ(u);
    qt = J(u + alpha*gradJ);
    q0 = J(u);
    dq0 = (gradJ.')*(gradJ);
    
    while(qt > q0 + m*dq0*alpha)
        alpha = alpha/2;
        qt = J(u - alpha*gradJ);
    end
    
    u = u - alpha*GJ(u);
    
    plot(u(1),u(2),'--c*','LineWidth',1)
    pause(0.2)
    title('EXO 3 : Aramijo')
    drawnow
                
end

hold off

%% Exercice 4 : Nelder et mead

J = @(u) (u(1,:)-1).^2 + 10*(u(1,:).^2 - u(2,:)).^2; % fonction rosenbrock
GJ = @(u) [2*(u(1,:)-1) + 2*10*(u(1,:).^2 - u(2,:)).*2.*u(1,:); -10*2*(u(1,:).^2 - u(2,:))];

%plot J
[U1, U2] =  meshgrid(-1.5:.05:1.5,-0.5:.05:1.5);
v1 = U1(:);
v2 = U2(:);
figure(4)
clf
contourf(U1, U2, reshape( J([v1'; v2']),size(U1,1), size(U1,2)))
title('EXO 4 : Nelder et Mead')
hold on

% initialize x
n=2;

% n+1 points affinements indépendants
x = zeros(n,n+1);
x(:,1) = [-1;0];
x(:,2:end) = (rand(n,n)).*([3;2]*ones(1,n)) - [1.5;0.5]*ones(1,n);
matx = x(:,1:end-1) - x(:,end)*ones(1,n);
while(rank(matx)~=n)
    x(:,2:end) = (rand(n,n)).*([3;2]*ones(1,n)) - [1.5;0.5]*ones(1,n);
    matx = x(:,1:end-1) - x(:,end)*ones(1,n);
end

plot([x(1,:), x(1,1)],[x(2,:), x(2,1)],'--*g','LineWidth',2)
drawnow
it_max = 100;
it = 1;
x1 = x(:,1);

diam = 100;

while(it <= it_max && diam>1e-5)
    % renuméroter les points tel que f(x1)<=...<=f(xn+1)
    [Jx, idx] = sort(J(x));
    
    x = x(:,idx);
        
    x1 = x(:,1);
    xn = x(:,end-1);
    x_last = x(:,end);
    
    %pause
    plot([x(1,:), x(1,1)],[x(2,:), x(2,1)],'--*r')
    drawnow
    hold on
    
    x_bar = mean(x(:,1:n),2);
    
    d = x_bar - x_last;
    
    xr = x_last + 2*d;
    
    if (J(xr)<J(x1))
        xe = x_last + 3*d;
        if(J(xe)<J(xr))
            x(:,end) = xe;
        else
            x(:,end) = xr;
        end
    else
        if ((J(x1)<=J(xr)) && (J(xr)<J(xn)))
            x(:,end) = xr;
        else
            if ((J(xn)<=J(xr)) && (J(xr)<J(x_last)))
                xc = x_last + 3/2*d;
                if (J(xc)<J(xr))
                    x(:,end) = xc;
                else
                    x(:,2:end) = x(:,2:end) + 1/2*(x(:,2:end)-x1*ones(1,n));
                end
            else
                xcc = x_last + 1/2*d;
                if (J(xcc)<J(xr))
                    x(:,end) = xcc;
                else
                    x(:,2:end) = x(:,2:end) + 1/2*(x(:,2:end)-x1*ones(1,n));
                end
            end
        end
    end
    
    it = it+1
    
    % simplex diameter
    diam = -100;
    for i =1:n
        for j=1:n
            dij = norm(x(:,i) - x(:,j));
            if dij > diam
                diam = dij;
            end
        end
    end
    
end

x(:,1)
plot([x(1,:), x(1,1)],[x(2,:), x(2,1)],'--*r')
hold off

%% Exercice 5 : Lenard Jones

V = @(r) 1./(r.^12) - 2./(r.^6);

N = 4;

n = 6;

X1 = [0 0 0];
X2 = [1 0 0];
X3 = [1/2 sqrt(3)/2 0];
X4 = [1/2 1/2/sqrt(3) sqrt(2/3)];

figure(5)
clf
plot3(X1(1), X1(2), X1(3), 'm*', 'LineWidth', 3)
title('EXO 5 : Lenard Jones')
hold on
plot3(X2(1), X2(2), X2(3), 'r*', 'LineWidth', 3)
plot3(X3(1), X3(2), X3(3), 'b*', 'LineWidth', 3)
plot3(X4(1), X4(2), X4(3), 'k*', 'LineWidth', 3)
grid on
xlabel('x')
ylabel('y')
zlabel('z')
%pause(5)

LJN(([1 1/2 sqrt(3)/2 1/2 1/2/sqrt(3) sqrt(2/3)].')*ones(1,7),V,N)

% initialize
% n+1 points affinements indépendants

% x = rand(n,n+1)*8-4;
% matx = x(:,1:end-1) - x(:,end)*ones(1,n);
% count_it = 0;
% while(rank(matx)~=n)
%     x = rand(n,n+1)*4-2;
%     matx = x(:,1:end-1) - x(:,end)*ones(1,n);
%     count_it = count_it +1
% end

x = ([1 1/2 sqrt(3)/2 1/2 1/2/sqrt(3) sqrt(2/3)].')*ones(1,7)+rand(n,n+1)*0.1;

XX = [0, x(1,1), x(2,1), x(4,1); 0, 0, x(3,1), x(5,1); 0, 0, 0, x(6,1)];

%plot3(XX(1,:),XX(2,:),XX(3,:),'--*g','LineWidth',2)
%xlabel('x')
%ylabel('y')
%zlabel('z')

it_max = 500;
it = 1;
x1 = x(:,1);

diam = 100;

while(it <= it_max && diam>1e-5)
    % renuméroter les points tel que f(x1)<=...<=f(xn+1)
    
    [Jx, idx] = sort(LJN(x,V,N));
    
    x = x(:,idx);
        
    x1 = x(:,1);
    xn = x(:,end-1);
    x_last = x(:,end);
    
    XX = [0, x(1,1), x(2,1), x(4,1); 0, 0, x(3,1), x(5,1); 0, 0, 0, x(6,1)];
    plot3(XX(1,:),XX(2,:),XX(3,:),'*','LineWidth',2,'color',rand(1,3))
    %axis([-2,2,-2,2,-2,2])
    drawnow
    %pause(5)
    
    x_bar = mean(x(:,1:n),2);
    
    d = x_bar - x_last;
    
    xr = x_last + 2*d;
    
    if (LJN(xr,V,N)<LJN(x1,V,N))
        xe = x_last + 3*d;
        if(LJN(xe,V,N)<LJN(xr,V,N))
            x(:,end) = xe;
        else
            x(:,end) = xr;
        end
    else
        if ((LJN(x1,V,N)<=LJN(xr,V,N)) && (LJN(xr,V,N)<LJN(xn,V,N)))
            x(:,end) = xr;
        else
            if ((LJN(xn,V,N)<=LJN(xr,V,N)) && (LJN(xr,V,N)<LJN(x_last,V,N)))
                xc = x_last + 3/2*d;
                if (LJN(xc,V,N)<LJN(xr,V,N))
                    x(:,end) = xc;
                else
                    x(:,2:end) = x(:,2:end) + 1/2*(x(:,2:end)-x1*ones(1,n));
                end
            else
                xcc = x_last + 1/2*d;
                if (LJN(xcc,V,N)<LJN(xr,V,N))
                    x(:,end) = xcc;
                else
                    x(:,2:end) = x(:,2:end) + 1/2*(x(:,2:end)-x1*ones(1,n));
                end
            end
        end
    end
    
    it = it+1
    
    % simplex diameter
    diam = -100;
    for i =1:n
        for j=1:n
            dij = norm(x(:,i) - x(:,j));
            if dij > diam
                diam = dij;
            end
        end
    end
    
end

hold off

XX = [0, x(1,1), x(2,1), x(4,1); 0, 0, x(3,1), x(5,1); 0, 0, 0, x(6,1)];
figure(6)
clf
plot3(XX(1,:),XX(2,:),XX(3,:),'*r','LineWidth',2)
xlabel('x')
ylabel('y')
zlabel('z')
grid on
% hold on
% plot3([-2 2], [0 0], [0 0], 'b', 'LineWidth', 1.5)
% plot3([0 0], [-2 2], [0 0], 'b', 'LineWidth', 1.5)
% plot3([0 0], [0 0], [-2 2],  'b', 'LineWidth', 1.5)
 %axis([-2,2,-2,2,-2,2])