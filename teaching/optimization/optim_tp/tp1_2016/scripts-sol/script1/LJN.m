function L = LJN(x,V,N)
%Energie potentielle totale du système constitué de N atomes dans R3

    XX = zeros(3,N,size(x,2));
    XX(1,2,:) = x(1,:);
    XX(1,3,:) = x(2,:);
    XX(1,4,:) = x(4,:);
    XX(2,3,:) = x(3,:);
    XX(2,4,:) = x(5,:);
    XX(3,4,:) = x(6,:);
    
    L = zeros(1,size(x,2));
    
    for i=1:N-1
        for j=i+1:N
            diff = reshape(XX(:,i,:)-XX(:,j,:), [size(XX,1),size(XX,3)]);
            nrm = sqrt(sum(diff.^2,1));
            L = L + V(nrm);
        end
    end

end