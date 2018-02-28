function L = LJN(u,N,V)
%Energie potentielle totale du système constitué de N atomes dans R3
   
L = 0;

for i=1:N-1
    for j=i+1:N
        v = u(:,i)-u(:,j);
        nrm = sqrt(v'*v);
        L = L + V(nrm);
    end
end

end