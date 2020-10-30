function nn = nnbp(nn)  
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights
% %反向传播
    
    n = nn.n;  %n=4
    m = size(nn.a{1},1);
    sparsityError = 0;
    
    switch nn.output       %输出层的误差
        case 'sigm'
         %   d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
          % d{n} =  ( -1./(sqrt(1-nn.e)) .* (y/ (norm(nn.a{n}) * norm(y)) - ( nn.a{n}.*(nn.a{n} .* y) )/ ( norm(nn.a{n}).^3 * norm(y)))).* (nn.a{n} .* (1 - nn.a{n}));
        d{n} =  nn.delt_y .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear','relu','lrelu'}
           %  d{n} = - nn.e;      %d{n}表示最后一层误差增量   目标函数对a{n}求偏导
           % d{n} = -1/sqrt(1-(dot(y, nn.a{n})/ (norm(nn.a{n}) * norm(y)).^2) ) .*  ( y/ norm(nn.a{n}) * norm(y) -  nn.a{n}.*(nn.a{n} .* y)/  norm(nn.a{n}).^3 * norm(y) ) ;
          %  d{n} = -1./(sqrt(1-nn.e)) .* (y/ (norm(nn.a{n}) * norm(y)) - ( nn.a{n}.*(nn.a{n} .* y) )/ ( norm(nn.a{n}).^3 * norm(y)));
        d{n} = nn.delt_y ;
    end
    
    for i = (n - 1) : -1 : 2
        
        if(nn.nonSparsityPenalty>0)
            Pi = repmat(nn.P{i}, size(nn.a{i}, 1), 1);
            sparsityError =  nn.nonSparsityPenalty * (-nn.sparsityTarget ./ Pi + (1 - nn.sparsityTarget) ./ (1 - Pi));
            %sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * ones(size(pi))];
            %sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives   %
        %if i+1==n % in this case in d{n} there is not the bias term to be removed         
         d{i} = d{i + 1} * nn.W{i} + sparsityError; % Bishop (5.56)   
       % else % in this case in d{i} the bias term has to be removed
       %     d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
       % end
          
        if(nn.dropoutFraction>0)
          d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end
        
        
        % Derivative of the activation function   求导
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
            case 'relu'
                d_act = zeros(size(nn.a{i}));
                d_act(nn.a{i}>0) = 1;  
               % d_act = (nn.a{i} > 0);
            case 'linear'
                d_act = ones(size(nn.a{i}));
            case 'lrelu'
               d_act = (nn.a{i} > 0) + nn.ra(i-1) * (nn.a{i} < 0);
                
              %  tt = d{i}(nn.a{i} < 0) .* nn.a{i}(nn.a{i}<0) / nn.ra(i-1);
               tt =  d{i}(nn.a{i} < 0) .* nn.a{i}(nn.a{i}<0) / nn.ra(i-1);
              % tt = tt(nn.a{i} < 0);
               nn.da(i-1) = sum(tt) / size(nn.a{i},1);
        end
         d{i} = d{i} .* d_act;  %dl/dy
      
     if nn.useBatchNormalization
            d_xhat = bsxfun(@times, d{i}, nn.gamma{i-1});
            x_mu = bsxfun(@minus, nn.a_pre{i}, nn.mu{i-1});
            inv_sqrt_sigma = 1 ./ sqrt(nn.sigma2{i-1} + nn.epsilon);
            d_sigma2 = -0.5 * sum(d_xhat .* x_mu) .* inv_sqrt_sigma.^3;
            d_mu = bsxfun(@times, d_xhat, inv_sqrt_sigma);
            d_mu = -1 * sum(d_mu) -2 .* d_sigma2 .* mean(x_mu);
            d_gamma = mean(d{i} .* nn.a_hat{i});
            d_beta = mean(d{i});
            di1 = bsxfun(@times,d_xhat,inv_sqrt_sigma);
            di2 = 2/m * bsxfun(@times, d_sigma2,x_mu);
            d{i} = di1 + di2 + 1/m * repmat(d_mu,m,1);
            nn.dBN{i-1} = [d_gamma d_beta];
            nn.d_sigma{i-1} = d_sigma2;
     end       
        
    end

    for i = 1 : (n - 1)
      %  if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);     %权值增量   
      %  else
         %   nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
     %   end
    end
end
