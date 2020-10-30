function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;     %n=4
    [m q] = size(x);
    
    % x = [ones(m,1) x];  % 增加1行值bias
    nn.a{1} = x;

    %feedforward pass   
    for i = 2 : n-1
        if nn.useBatchNormalization
            if nn.testing
                nn.a_pre{i} = nn.a{i - 1} * nn.W{i - 1}';
                norm_factor = nn.gamma{i-1}./sqrt(nn.mean_sigma2{i-1}+nn.epsilon);
                nn.a_hat{i} = bsxfun(@times, nn.a_pre{i}, norm_factor);
                nn.a_hat{i} = bsxfun(@plus, nn.a_hat{i}, nn.beta{i-1} -  norm_factor .* nn.mean_mu{i-1});
            else
                nn.a_pre{i} = nn.a{i - 1} * nn.W{i - 1}';
                nn.mu{i-1} = mean(nn.a_pre{i});
                x_mu = bsxfun(@minus,nn.a_pre{i},nn.mu{i-1});
                nn.sigma2{i-1} = mean(x_mu.^2);
                norm_factor = nn.gamma{i-1}./sqrt(nn.sigma2{i-1}+nn.epsilon);
                nn.a_hat{i} = bsxfun(@times, nn.a_pre{i}, norm_factor);
                nn.a_hat{i} = bsxfun(@plus, nn.a_hat{i}, nn.beta{i-1} -  norm_factor .* nn.mu{i-1});
            end
        else
            nn.a_hat{i} = nn.a{i - 1} * nn.W{i - 1}';
        end 
        
        switch nn.activation_function 
            case 'sigm'
                nn.a{i}=sigm(nn.a_hat{i});
           case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a_hat{i});
            case 'relu'
                nn.a{i} = max(nn.a_hat{i},0);
            case 'lrelu'
                nn.a{i} = max(nn.a_hat{i},0 )+ nn.ra(i-1) * min(nn.a_hat{i}, 0);
        end 
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        %change Sparsitypara 0.1
        if(nn.nonSparsityPenalty>0)
            nn.P{i} = 0.9 * nn.P{i} + 0.1 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
       %  nn.a{i} = [ones(m,1) nn.a{i}];
      % nn.a{i} = nn.a{i};
     
    end
    
    % 隐藏层最后一层先进行正则向，非负约束
    
    %最后一层输出
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
         case 'relu'
            nn.a{n} = max(nn.a{n-1} * nn.W{n - 1}',0);
            %nn.a{i} = max(nn.a_hat{i},0+ nn.ra(i-1) * min(nn.a_hat{i}, 0));
            %nn.a{n} = max(nn.a{n-1} * nn.W{n - 1}',0.01*(nn.a{n-1} * nn.W{n - 1}'));
         case 'lrelu'
           nn.a{n} = max(nn.a{n-1} * nn.W{n - 1}',0);
    end

    %error and loss

 nn.e = acos( sum(y.*nn.a{n},2)./sqrt(sum(nn.a{n}.*nn.a{n},2)) ./ sqrt(sum(y.*y,2)))/pi;

 nn.delt_y =(1/pi).*(nn.a{n}.* sum(y.*nn.a{n},2)-y.*sum(nn.a{n}.*nn.a{n},2))./sqrt(sum(nn.a{n}.*nn.a{n},2)).^3./(sqrt(sum(y.*y,2)))./sqrt(1-(sum(y.*nn.a{n},2)./sqrt(sum(nn.a{n}.*nn.a{n},2)) ./ sqrt(sum(y.*y,2))).^2);
  %  rho = (1/q).*sum(nn.a(n),2); (y.*nn.a{n})
   % Jsparse = sum(nn.nonSparsityPenalty.*log(nn.nonSparsityPenalty./rho)+ ...
    %    (1-nn.nonSparsityPenalty).*log((1-nn.nonSparsityPenalty)./(1-rho)));
    switch nn.output
        case {'sigm', 'linear','relu','lrelu'}
           % nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
            nn.L =  sum(nn.e)/m ;    % dot矩阵每一列相乘之和 ， norm求出的是固定值
        case 'softmax'
              %error and loss change y at nn.e
            nn.L = -sum(sum( nn.e .* log(nn.a{n}))) / m;
    end
   
    
end
