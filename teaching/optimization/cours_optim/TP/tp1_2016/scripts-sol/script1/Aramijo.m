%%%% Regle d'Armijo

clear all;

J = @(u) (u(1,:)-1).^2 + 100*(u(1,:).^2 - u(2,:)).^2; % fonction rosenbrock
GJ = @(u) [2*(u(1,:)-1) + 2*(u(1,:).^2 - u(2,:)).*2.*u(1,:); -100*2*(u(1,:).^2 - u(2,:))];
dq0 = @(u,gj) [ (2*(u(1,:)-1).*(-gj(1,:)) + 100*2*(u(1,:).^2-u(2,:)).*(2*u(1,:).*(-gj(1,:))+gj(2,:))).*(-gj(1,:)); (2*(u(1,:)-1).*(-gj(1,:)) + 100*2*(u(1,:).^2-u(2,:)).*(2*u(1,:).*(-gj(1,:))+gj(2,:))).*(-gj(2,:))];
rhok = @(R,k) R/(1 + (k/10));

R = 1;
max_it = 100;
m = 0.1;

rho0 = R;
u0 = [-1; 0];

u = u0;

%plot J
[U1 U2] = meshgrid(-1.5:.01:2,-0.5:.01:3);
v1 = U1(:);
v2 = U2(:);
%mesh(U1, U2, reshape( J([v1'; v2']),size(U1,1), size(U1,2)) )
contourf(U1, U2, reshape( J([v1'; v2']),size(U1,1), size(U1,2)) )

hold on

%trajectoire
plot(u(1),u(2),'r*')

% initial alpha value
alpha = rho0;
while(1)
    if (J(u - alpha*GJ(u))) <= J(u) + m*alpha*dq0(u,GJ(u))
        break;
    else
        alpha = alpha/2;
    end    
end

%calcul de alphak
for k = 1:max_it

    u = u - alpha*GJ(u)
    
    
    %compute new alpha
    alpha = rhok(R,k)
    while(1)
        if (J(u - alpha*GJ(u))) <= J(u) + m*alpha*dq0(u,GJ(u))
            break;
        else
            alpha = alpha/2;
        end    
    end
    
    plot(u(1),u(2),'or')
    
end

hold off