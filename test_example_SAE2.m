
%%  mnist data
clear all; close all; clc;
%% 

inputFilename = 'H:\a. deeplearningtest\SAE_realdata\unmixing_mineral_SAE_cup\Cuprite_S1_R188.img';
h = 250;
w = 190;
p = 188;
N = w*h;

M = multibandread(inputFilename, [h w p], 'int16', 0, 'bsq', 'ieee-le')/1e4; % 读取的高光谱三维数据
%x = M(31:220,:,:); % 变为方阵    从31到230每隔一排取值
% load trainlabel.mat
traindata = hyperConvert2d(M);

traindata = traindata(:,2:188);
tt = traindata';
%% 光谱平滑   
train_data = smooth(tt,5);
traindata1 = reshape(train_data,187, 47500 );
traindata2 = traindata1';

train_x = double(traindata2);
%test_x  = double(traindata1);
train_y = double(traindata2);
%test_y  = double(traindata1);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0);

sae = saesetup([187 120 60 14]);   %网络结构建立    sae元胞矩阵       
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.01;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
sae.ae{1}.nonSparsityPenalty =0.3;  %
sae.ae{1}.output                    = 'sigm';

sae.ae{2}.activation_function       = 'sigm';                         
sae.ae{2}.learningRate              = 0.01;
sae.ae{2}.inputZeroMaskedFraction   = 0; %这里的denoise autocoder相当于隐含层的dropout,但它是分层训练的
sae.ae{2}.nonSparsityPenalty =0.2;
sae.ae{2}.output                    = 'sigm';

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 0.1;
sae.ae{3}.inputZeroMaskedFraction   = 0; %这里的denoise autocoder相当于隐含层的dropout,但它是分层训练的
sae.ae{3}.nonSparsityPenalty =0.2;

opts.numepochs =  100;
opts.batchsize = 50;

%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([187 120 60 14 187]);   %构建网络结构
nn.activation_function               =  'sigm';
nn.learningRate                      = 0.02;
nn.output = 'relu';

%nn.W{3} = sae.ae{3}.W{1};

%sae = saetrain(sae, train_x, opts);   %网络训练
%nn.W{1} = sae.ae{1}.W{1};
%nn.W{2} = sae.ae{2}.W{1};
% Train the FFNN

opts.numepochs =   100;
opts.batchsize = 50;
%标签数据微调
[nn,L] = nntrain(nn, train_x, train_y, opts);

%%
%光谱角误差
aa1 =  xlsread('endmember.xlsx',1,'B4:BA190');   % usgs重采样 187个波段
aa2 = smooth(aa1,5);
aa = reshape(aa2,187, 52 );
% aa = aa(88:end,:);
% aa =  xlsread('endm_cup.xlsx',3,'B3:M189');
% aa = aa([4:29 33:93 95:187],:);
bb = nn.W{1,4};
%bb= bb(2:188,:);
ll = correspond_end(bb,aa);
% cc = bb(:,ll);    

% 从若干端元中挑选匹配最高的
cc = aa(:,ll);   % 端元

% SAM_err = acos( sum(aa.*cc)./sqrt(sum(cc.*cc)) ./ sqrt(sum(aa.*aa)));

SAM_err = acos( sum(bb.*cc)./sqrt(sum(cc.*cc)) ./ sqrt(sum(bb.*bb)));

SAM_mean = mean(SAM_err);

%%