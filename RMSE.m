function f=RMSE(M_1abel,M_result)
c=0;
[m,n] = size(M_1abel);
for i=1:m
    for j=1:n
        w=M_1abel(i,j)-M_result(i,j);
        c=c+w^2;
    end
end
f=sqrt(c/(m*n));