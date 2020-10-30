function [er, bad,labels] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
   
    % [dummy, expected] = max(y,[],2);
    
   %  bad = find(labels ~= expected);    
   %  er = numel(bad) / size(x, 1);
    % all mineral RMSE
    er = RMSE(y,labels);
    bad = find(er > 0.1);  
   % corr2(y,labels);
    
    fprintf('Total Accuracy: %0.3f%%\n', er * 100);

end
