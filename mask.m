clear
clc
close all

n = 500; %numero de datos
d = 200; %dimension de datos
e = 0.2; % epsilon

resto = floor((1/e) - mod(floor(log(n)/e^2),(1/e)));

m = floor(log(n)/e^2) + resto; % Reduccion de dimension
s = floor(m*e);  % Numero de bloques

p = ones(m,d)*e;  % Matriz p inicializada con epsilons en todas sus posiciones
Bq = reshape(1:m,1/e,s); %posiciones de los bloques Bq = {[1,...,1/e],[1/e+1,...,2/e]....}

p(Bq(:,1),1)=zeros(length(Bq(:,1)),1);
p(Bq(1,1),1)=1; %%%%%%%% Arreglar, primera iteracion!!!, se pone el primer bloque de p en [1 0 0 .... 0]
tic
for q = 1:s % bloque q hasta s
    
    if q==1     % Arreglo para la primera iteracion, como ya se inicializo
        ini1=2; % el primer bloque para q=1, l debe empezar en 2
    else
        ini1=1; % pero para las siguientes filas de bloques, l regresa a 1.
    end
    
    for l = ini1:d % columna l -> bloque (q,l)
        
        %% Sumatoria alfas
        alfa = zeros(length(Bq(:,1)),1); % vector donde se guardaran los estimadores pesismistas y donde se encontrara el minimo
        rr = 1:1/e; % Este es el r que queremos optimizar en alfa
        
        for j = 1:l-1 % sumatoria alfa
            
            %% productoria D            
            if q==1 % primeras filas de bloques, esto era lo que te
                ini2=0; %decia de que si es la primera fila de bloques
            else        % no se cumple que q_bar<q, por lo que se vuelve un unico producto
                ini2=1;
            end
            
            q_bar = 1:q-ini2;
            dum = reshape(p(Bq(:,q_bar),j).*p(Bq(:,q_bar),l),1/e,q-ini2);
            rho = sum(dum);
            D = prod(1+rho); %productoria de D
            
            alfa = alfa+(D*p(Bq(rr,q),j)); %sumatoria de los j<k
        end
        
        [~,I] = min(alfa); % estimador minimo
        p(Bq(:,q),l) = 0;  % asignacion 1
        p(Bq(I,q),l) = 1;  % asignacion 0s
%         disp([q,l])
    end
end
toc