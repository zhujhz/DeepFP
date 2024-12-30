%% demo reshape channel 
clear
clc
M = 256; N = 8; K = 24; 
H = zeros(N,M,K);
load("layout_3cell_MIMO_K24.mat");
for k = 1:K
    H(:,1:80,k) = c(1,k).coeff(:,:,1,1);
    H(:,81:160,k) = c(1,k+K).coeff(:,:,1,1);
    H(:,161:256,k) = c(1,k+2*K).coeff(:,:,1,1);
end
save('channel_K24','H')