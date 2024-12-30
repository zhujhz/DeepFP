close all
clear all
clc
set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.7])            	% Default Paper Size
rand('seed',1)

N_t=256;
N_t_1=86;
N_t_2=86;
N_t_3=84;

N_r=8;
User_number=12;
N_samples=5000;


H = zeros(N_r,N_t,User_number,N_samples);
% load("layout_3cell_MIMO_K24.mat");
for n=1:N_samples
    % n
s = qd_simulation_parameters;
s.center_frequency  = [6.7e9];                          % Assign two frequencies
s.samples_per_meter = 1200; 
s.use_3GPP_baseline = 1; 
s.use_random_initial_phase = 0;
l = qd_layout(s);                                     % New QuaDRiGa layout

%% Base Station
l.no_tx = 3;
l.tx_position = [10 0 25; -10 0 25;0 10 25]'; %Location
%% Cell 3
l.no_rx = User_number;                                      
for rx = 1:l.no_rx
    position = [ ...
                        -250 + 500*rand(1); ...
                        -350 + 700*rand(1); ...
                        1.5; ...
                    ];
    motion_direction = 2 * pi * rand;
    track_len = 1/60;
    t = qd_track('linear', track_len, motion_direction);
    t.initial_position = position;
    t.set_speed(3/3.6);
    l.rx_track(1, rx) = t;    
    l.rx_track(1, rx).initial_position = t.initial_position;
    l.rx_track(1, rx).name = ['Rx' num2str(rx)];
    t.interpolate_positions(s.samples_per_meter);
end
sample = 21;
set(0,'DefaultFigurePaperSize',[14.5 7.7])              % Adjust paper size for plot
% l.visualize([],[],0);                                   % Plot the layout
% view(-33, 60);                                          % Enable 3D view



%% Antenna set-up 256 = 16 * 16
a_3500_Mhz5  = qd_arrayant( '3gpp-3d', N_t_1,1, s.center_frequency,1);
a_3500_Mhz6  = qd_arrayant( '3gpp-3d', N_t_3,1, s.center_frequency,1);
l.tx_array(1,1) = copy(a_3500_Mhz5);
l.tx_array(1,2) = copy(a_3500_Mhz5);
l.tx_array(1,2).rotate_pattern(120,'z');
l.tx_array(1,3) = copy(a_3500_Mhz6);
l.tx_array(1,3).rotate_pattern(240,'z');
%% User
a_35_Mhz  = qd_arrayant( '3gpp-3d',N_r,1,s.center_frequency,1);
for i = 1 : l.no_rx
    l.rx_array(1,i) = copy(a_35_Mhz);
end
%% Generate channel coefficients
% Channel coefficients are generated by calling "l.get_channels". The output is an array of QuaDRiGa
% channel objects. The first dimension corresponds to the MTs (100). The second dimension
% corresponds to the number of BSs (1) and the third dimension corresponds to the number of
% frequencies (2).
l.set_scenario('3GPP_38.901_UMa_LOS');                    % 34 paths  
p = l.init_builder;   
for i = 1:l.no_tx
    p(i).scenpar.NumClusters = 1;                      % Only generate 5 clusters
end
p.gen_parameters;                                        % Generate small-scale-fading parameters
c = p.get_channels;  
% save('layout_3cell_MIMO_K24')

% H = zeros(N_r,N_t,User_number);
% load("layout_3cell_MIMO_K24.mat");
for k = 1:User_number
    H(:,1:N_t_1,k,n) = c(1,k).coeff(:,:,1,1);
    H(:,N_t_1+1:N_t_1+N_t_2,k,n) = c(1,k+User_number).coeff(:,:,1,1);
    H(:,N_t_1+N_t_2+1:N_t,k,n) = c(1,k+2*User_number).coeff(:,:,1,1);
end


end

save('channel_K24_256_8_12_5000.mat','H')