function labels = nnpredict(nn, x)  %Ԥ����������
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
   
    %�޸�label ��ɿ��ﺬ��
   % [dummy, i] = max(nn.a{end},[],2);  %�������ԭ��ȷ���������࣬��ȡ����ǩ
    labels = nn.a{end};
    
end
