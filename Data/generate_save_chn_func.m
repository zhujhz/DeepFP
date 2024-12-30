function [chn] = generate_save_chn_func(N_t,N_r,K,B,N_samples)

idx=1;

% for idx=[1,2,3,4,5,6,7,8,9,10]
% N_t=64;
% N_r=4;
usernumber=K;
BSnumber=B; 
total_usernumber=BSnumber*usernumber;
w = ones(usernumber,BSnumber)/usernumber; 

Factor = 1e6;
bandwidth = 10;
noise = 10^((-169-30)/10)*Factor;
numSlot = 1;
numTone = 1;
% maxPower = ones(1,numBS)*10^((-47-30)/10)*Factor;


noise_power=1e-8;
mimoPattern = [N_t,N_r]; % [tx,rx]


chn=zeros(N_samples,N_r,N_t,usernumber,BSnumber,BSnumber);
for i=1:N_samples
    [generate_chn, distPathLoss] = GenerateNetwork7(bandwidth, BSnumber, total_usernumber,mimoPattern,numTone);
    generate_chn =permute(generate_chn, [1 2 4 3 ,5]);
    for j = 1:BSnumber
        for k = 1:usernumber
            chn(i, :,:, k, j, :) = generate_chn(:, :, k + (j-1)*usernumber, :,:);
        end
    end
end



fileName = [num2str(idx) '_' num2str(N_samples) '_Nr_' num2str(N_r) '_Nt_' num2str(N_t) '_N_user_' num2str(usernumber) '_BS_' num2str(BSnumber) '.mat'];
% 保存数据
save(fileName, 'chn');
end