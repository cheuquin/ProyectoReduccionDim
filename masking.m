% function masking

n = 50;
d = 1000;

e = 0.1;

a = 0; b = 0.5;
V = (b-a).*rand(n,d) + a;
V(V<=2*b/3) = 0;

s = ceil(log(n)/e);
m = s/e;
t = 3/(2*log(2)*m);

psi = zeros(n,m);
for i = 1:n
    psi(i,1) = exp(-s^2*t*log(2) + s*log(1+e)*(1-norm(V(i,:),4)^4)/2);
end

% psi = sum(psi);
Bq = reshape(1:m,1/e,s);

p = ones(m,d)*e;
b = ones(m,1);

for q = 1:s
    nu = zeros(n,1);
    psir = psi(Bq(:,q));
    for j = 1:d
        k  = 1;
        for i = 1:n%1/e
            if V(i,j)~=0
                while k<j  % k denota columna
                    if V(i,k)~=0
                        rk = Bq(p(Bq(:,q),k)==1,q);
                        if b(rk)==0
                            b(rk)=1;
                        end
                        psi(i,rk)=psi(i,rk)*2^(V(i,k)^2*V(i,j)^2);
                    end
                    k = k+1;
                end
            end
        end
        for r = 1:m
            mm=0;
            if b(r) == 1
                for i= 1:n
                    if V(i,j)~=0
                        mm = mm + psi(i,r)*(1+e)^(-nu(i)*V(i,j)^2);
                    end
                end
                psir(r) = mm;
            end
        end
        r = Bq(:,q);
        r = r(b(r)==1);
        [Y,I] = max(psir(r));
        r_a = r(I);
        
        p(r,j) = 1;
        mm = Bq(:,q);
        mm(mm == r) = [];
        p(mm,j) = 0;
        
        for i = 1:n
            if V(i,j)~=0
                nu(i) = nu(i)+V(i,j)^2;
            end
        end
        
        for r = 1:m
            if b(r) == 1
                psi(i,r) = 1;
                b(r) = 0;
            end
        end
    end
end

% end