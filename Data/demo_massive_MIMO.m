clear
clc
Factor = 1e6;
bandwidth = 10;
numBS = 1;
numUser = 2*3*1; % Q*L num sector * numUserin each sector
noise = 10^((-169-30)/10)*Factor;
numSlot = 1;
maxPower = ones(1,numBS)*10^((-47-30)/10)*Factor;

numTone = 1;
L = numBS; Q = 6; w = ones(Q,L)/Q; 
max_iter = 2*1e1; % max iteration of CQT
sigma = 1e-8; M = 128; N = 4; Pmax = 1e2;
test_num = 100; % number of random tests
iter_times = 5; % the max iteration of NQT and EQT is iter_times*max_iter
mimoPattern = [M,N]; % [tx,rx]
All_iter_results1 = zeros(max_iter+1,test_num);
All_time1 = zeros(max_iter,test_num); 

All_iter_results2 = zeros(iter_times*max_iter+1,test_num);
All_time2 = zeros(iter_times*max_iter,test_num);

All_iter_results3 = zeros(iter_times*max_iter+1,test_num);
All_time3 = zeros(iter_times*max_iter,test_num);

All_iter_results4 = zeros(iter_times*max_iter+1,test_num);
All_time4 = zeros(iter_times*max_iter,test_num);

All_iter_results5 = zeros(iter_times*max_iter+1,test_num);
All_time5 = zeros(iter_times*max_iter,test_num);

load("1_10000_Nr_4_Nt_128_N_user_6_BS_1.mat")
chna=chn;
for test = 1:test_num
    
    % [chn, distPathLoss] = GenerateNetwork7(bandwidth, numBS, numUser,mimoPattern,numTone);
    % % chn = squeeze(chn);
    % chn =permute(chn, [1 2 4 3]);

    % real_part = randn(4,128,6,1);  % 实部，标准正态分布
    % imag_part = randn(4,128,6,1);  % 虚部，标准正态分布
    % 
    % % 组合为复数矩阵
    % chn = real_part + 1i * imag_part;

    % chn = permute(chn, [1, 2, 3, 4]);
    % V = Generate_V(M,Q,L,Pmax); % Randomly initialize V
    chn=chna(1,:,:,:,:);
    chn =permute(chn, [2 3 4 1]);
    V =sqrt(100/6)*ones(M,Q,L);
    % [All_iter_results1(:,test),  All_time1(:,test)] = CQT(max_iter,sigma,M,N,L,Q,chn,w,Pmax,V);
    [All_iter_results2(:,test),  All_time2(:,test)] = NQT(iter_times*max_iter,sigma,M,N,L,Q,chn,w,Pmax,V,false);
    [All_iter_results3(:,test),  All_time3(:,test)] = EQT(iter_times*max_iter,sigma,M,N,L,Q,chn,w,Pmax,V,false);

    % [All_iter_results4(:,test),  All_time4(:,test)] = NQT(iter_times*max_iter,sigma,M,N,L,Q,chn,w,Pmax,V,true);
    % [All_iter_results5(:,test),  All_time5(:,test)] = EQT(iter_times*max_iter,sigma,M,N,L,Q,chn,w,Pmax,V,true);
    test
end

ave_iter1 = sum(All_iter_results1(:,1:test_num),2)/test_num;
ave_time1 = sum(All_time1(:,1:test_num),2)/test_num;
% plot(ave_iter1)
% hold on

ave_iter2 = sum(All_iter_results2(:,1:test_num),2)/test_num;
ave_time2 = sum(All_time2(:,1:test_num),2)/test_num;
plot(ave_iter2)
hold on

ave_iter3 = sum(All_iter_results3(:,1:test_num),2)/test_num;
ave_time3 = sum(All_time3(:,1:test_num),2)/test_num;
plot(ave_iter3)

% ave_iter4 = sum(All_iter_results4(:,1:test_num),2)/test_num;
% ave_time4 = sum(All_time4(:,1:test_num),2)/test_num;
% plot(ave_iter4,'--')
% 
% 
% ave_iter5 = sum(All_iter_results5(:,1:test_num),2)/test_num;
% ave_time5 = sum(All_time5(:,1:test_num),2)/test_num;
% plot(ave_iter5,'--')


% 添加图例
legend('Nonhomogeneous FP', 'Extrapolated FP','Nonhomogeneous FP New', 'Extrapolated FP New');

hold off



