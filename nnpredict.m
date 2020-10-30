function labels = nnpredict(nn, x)  %预测样本分类
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
   
    %修改label 变成矿物含量
   % [dummy, i] = max(nn.a{end},[],2);  %依据最大化原则确定样本分类，提取类别标签
    labels = nn.a{end};
    
end
