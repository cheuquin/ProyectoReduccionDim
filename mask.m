n = 50;
d = 50;
e = 0.1;

s = ceil(log(n)/e);
m = ;
t = 3/(2*log(2)*m);

p = ones(m,d)*e;
Bq = reshape(1:m,1/e,s);

p(Bq(:,1),1)=[1 zeros(length(Bq(:,1))-1,1)']; %%Arreglar, primera iteracion

for q = 1:s % bloque q
    for l = 2:d % col l -> bloque (q,l) UNIFORME
        
        %% Sumatoria alfas
        alfa = zeros(length(Bq(:,1)),1);
        for rr = 1:1/e
            a = 0;
            for j = 1:l-1 % sumatoria alfa
                
                %% productoria D
                D = 1;
                for q_bar = 1:q-1
                    rho = sum(p(Bq(:,q_bar),j).*p(Bq(:,q_bar),k));
                    D = D*(1+rho);
                    
                end
                a = a+(D*p(Bq(rr,q),j));
            end
            alfa(rr) = a;
        end
        [~,I] = min(alfa);
        p(Bq(:,q),l) = 0;
        p(Bq(I,q),l) = 1;
    end
end