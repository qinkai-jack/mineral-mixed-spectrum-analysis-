function dd1 = BatchNormalizepass( xx, u,v)
m = size(xx);
epsilon = 10^(-9);
 % y_u = -1/(v+epsilon);
% y_v = (xx - u)*(-1/2).* (v+epsilon)^(-2/3);
% v_u = sum(-2*(xx-u)/m);
% y_x = 1/sqrt(v+epsilon);
v_x = 2*(xx - u)/m;
u_x = 1/m;
dd1 = (1-u_x)./sqrt(v+epsilon ) - (xx-u)./(v+epsilon).^(3/2)/2 .*v_x;

end


