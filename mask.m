n = 400;
d = 50;
e = 0.1;


resto = ceil((1/e) - ceil(log(n)/e^2));

m = ceil(log(n)/e^2) + resto;
s = ceil(m*e);
t = 3/(2*log(2)*m);

p = ones(m,d)*e;
Bq = reshape(1:m,1/e,s);

p(Bq(:,1),1)=zeros(length(Bq(:,1)),1); %%Arreglar, primera iteracion
p(Bq(1,1),1)=1;

for q = 1:s % bloque q
    for l = 2:d % col l -> bloque (q,l) UNIFORME MIRAR INICIAL
        
        %% Sumatoria alfas
        alfa = zeros(length(Bq(:,1)),1);
        for rr = 1:1/e
            a = 0;
            for j = 1:l-1 % sumatoria alfa
                
                %% productoria D
                D = 1;
                
                if q==1 % primeras filas de bloques
                    sust=0;
                else
                    sust=1;
                end
                
                for q_bar = 1:q-sust
                    rho = sum(p(Bq(:,q_bar),j).*p(Bq(:,q_bar),l));
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