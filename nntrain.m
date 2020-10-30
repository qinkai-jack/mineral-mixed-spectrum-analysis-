function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e1               = [];
%loss.val.e1                 = [];
%loss.value.e1         =  [];

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
loss.val.er                 = [];

loss.train.delt_y          =  [];
%loss.value.delt_y          =  [];
%loss.value.delt_y_frac          =  [];
opts.validation = 0;

if nargin == 6      %�����������
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);   %��

batchsize = opts.batchsize;%ÿ��batch����������
numepochs = opts.numepochs;%��������ѵ������
numbatches = m / batchsize;     %�ж���batch


assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs   
    tic;
     for j = 2 : nn.n
        nn.mean_sigma2{j-1} = 0;
        nn.mean_mu{j-1} = 0;
     end
    
    kk = randperm(m);  %��1��M�����
    for l = 1 : numbatches    %ÿ����ȡbatch�� ������һ����ȡnumepochs��
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);  %�����ȡ1��batchsize������
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);   %ǰ������
        nn = nnbp(nn);     %����Ȩֵ�ݶ�
        nn = nnapplygrads(nn);   %��������
        
        L(n) = nn.L;
      for j = 2 : nn.n
            nn.mean_sigma2{j-1} = nn.mean_sigma2{j-1} + nn.sigma2{j-1};
            nn.mean_mu{j-1} = nn.mean_mu{j-1} + nn.mu{j-1};
      end    
        n = n + 1;
%%        
  %      if mod(l,10)==0
  %          fprintf('epoch:%d iteration:%d/%d\n',i,l,numbatches);
  %          gradientNorm = [];
  %          for ll = 1:nn.n-1
  %              gradientNorm = [gradientNorm ' ' num2str(norm(nn.dW{ll}(:,2:end)))];
  %          end
  %          disp(gradientNorm);
  %      end
  %      if mod(l,100)==0
  %          disp([nn.ra mean(nn.gamma{2}) mean(nn.beta{2})]);
  %      end     
    end
    
     
    for j = 2 : nn.n
        nn.mean_sigma2{j-1} = nn.mean_sigma2{j-1} / (numbatches - 1);
        nn.mean_mu{j-1} = nn.mean_mu{j-1} / numbatches;
    end 
    t = toc;

    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;   %����ѧϰ��
    
     if nn.learningRate < 0.00001
        nn.learningRate = 0.00001;
     end
    
    
end
end

