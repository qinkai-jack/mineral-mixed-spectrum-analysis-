function xx = BatchNormalize(x,u,v)
epsilon = 10^(-9);
xx =(x-u)./sqrt(v+epsilon);
% xx = diag(v+epsilon)^(-0.5) *(x-u)
end
