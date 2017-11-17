clear
clc
close all

n = 500; %numero de datos
d = 200; %dimension de datos
e = 0.2; % epsilon

resto = ceil((1/e) - mod(ceil(log(n)/e^2),(1/e)));

m = ceil(log(n)/e^2) + resto; % Reduccion de dimension
s = ceil(m*e);  % Numero de bloques

p = ones(m,d)*e;  % Matriz p inicializada con epsilons en todas sus posiciones
Bq = reshape(1:m,1/e,s); %posiciones de los bloques Bq = {[1,...,1/e],[1/e+1,...,2/e]....}

p(Bq(:,1),1)=zeros(length(Bq(:,1)),1); 
p(Bq(1,1),1)=1; %%%%%%%% Arreglar, primera iteracion!!!, se pone el primer bloque de p en [1 0 0 .... 0]

for q = 1:s % bloque q hasta s
    
    if q==1     % Arreglo para la primera iteracion, como ya se inicializo 
        ini1=2; % el primer bloque para q=1, l debe empezar en 2
    else
        ini1=1; % pero para las siguientes filas de bloques, l regresa a 1.
    end
    
    for l = ini1:d % columna l -> bloque (q,l) 
        
        %% Sumatoria alfas
        alfa = zeros(length(Bq(:,1)),1); % vector donde se guardaran los estimadores pesismistas y donde se encontrara el minimo
        for rr = 1:1/e % Este es el r que queremos optimizar en alfa
            a = 0;
            for j = 1:l-1 % sumatoria alfa
                
                %% productoria D
                D = 1;
                
                if q==1 % primeras filas de bloques, esto era lo que te 
                    ini2=0; %decia de que si es la primera fila de bloques
                else        % no se cumple que q_bar<q, por lo que se vuelve un unico producto
                    ini2=1;
                end
                
                for q_bar = 1:q-ini2 % sumatoria de Rho
                    rho = sum(p(Bq(:,q_bar),j).*p(Bq(:,q_bar),l));
                    D = D*(1+rho); %productoria de D
                    
                end
                a = a+(D*p(Bq(rr,q),j)); %sumatoria de los j<k
            end
            alfa(rr) = a; % estimador pesimista para la posicion r en Bq
        end
        [~,I] = min(alfa); % estimador minimo
        p(Bq(:,q),l) = 0;  % asignacion 1
        p(Bq(I,q),l) = 1;  % asignacion 0s
        disp([q l])
    end
end