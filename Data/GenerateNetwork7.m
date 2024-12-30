function [ chn, distPathLoss ] = GenerateNetwork7( BW, L, K, mimo_pattern, T,R ,stand)

% Downlink channel
% chn(numRxAnte, numTxAnte, tone, user, BS)
if nargin < 6
        R = 0.8; % 如果没有提供第三个参数，使用默认值
        stand=8;
end

MACRO_MACRO_DIST = R; % macro-macro distance, in km (1.0)
% MACRO_MACRO_DIST = 1.6; % macro-macro distance, in km (1.0)

NUM_SECTOR = 3;
cell_radius = MACRO_MACRO_DIST/sqrt(3); % radius of macro cell

% macro-BS locations
BS_loc_all = [
    0
    exp(pi*1i/6)
    exp(-pi*1i/6)
    exp(-pi*1i/2)
    exp(pi*7i/6)
    exp(pi*5i/6)
    exp(pi*1i/2)
    ]*MACRO_MACRO_DIST;
BS_loc=BS_loc_all(1:L);

% MS locations
num_MS_per_cell = K/L;
num_MS_per_sector = num_MS_per_cell/NUM_SECTOR;

MS_loc = NaN(K,1);
for m = 0:L-1
    for s = 0:NUM_SECTOR-1
        for u = (1:num_MS_per_sector) + m*num_MS_per_cell...
                + s*num_MS_per_sector
            while 1
                x = 3/2*cell_radius*rand(1)-cell_radius/2;
                y = sqrt(3)/2*cell_radius*rand(1);
                if (y+sqrt(3)*x>0) && (y+sqrt(3)*x-sqrt(3)*cell_radius<0)...
                        && abs(x+1i*y)>.3*(cell_radius/0.8)
                    MS_loc(u) = (x+1i*y)*exp(1i*2*pi/3*s) + BS_loc(m+1);
                    break
                end
            end
        end
    end
end

% plot the topology
% figure; hold on;
% plot(real(BS_loc(1:L)), imag(BS_loc(1:L)),'r+');
% plot(real(MS_loc(1:K)), imag(MS_loc(1:K)),'bo');
% axis([-1 1 -1 1]*1.5); legend('Macro BS','MS');
% xlabel('km'); ylabel('km');

% % 绘制图形
% % figure; hold on;
% % 六边形蜂窝参数
% hex_radius = 0.8 / sqrt(3); % 半径，基站间距为 0.8
% theta = linspace(0, 2 * pi, 7); % 六边形的角度
% hexagon_x = hex_radius * cos(theta); % 六边形 x 坐标
% hexagon_y = hex_radius * sin(theta); % 六边形 y 坐标
% 
% % 每个基站用户的颜色
% colors = lines(length(BS_loc)); % 使用 'lines' 颜色方案，为每个基站设置不同颜色
% 
% % 分配用户到最近的基站
% user_assignment = zeros(length(MS_loc), 1); % 初始化用户分配
% for i = 1:length(MS_loc)
%     distances = abs(MS_loc(i) - BS_loc); % 计算用户到每个基站的距离
%     [~, closest_bs] = min(distances); % 找到最近的基站
%     user_assignment(i) = closest_bs; % 将用户分配给该基站
% end
% 
% % 绘制图形
% figure; hold on;
% 
% % 绘制每个基站的蜂窝六边形
% for k = 1:length(BS_loc)
%     center_x = real(BS_loc(k));
%     center_y = imag(BS_loc(k));
%     plot(center_x + hexagon_x, center_y + hexagon_y, 'k-', 'LineWidth', 1); % 六边形边界
% end
% 
% % 绘制基站位置
% % plot(real(BS_loc), imag(BS_loc), 'r^', 'MarkerSize', 10, 'DisplayName', 'Macro BS');
% plot(real(BS_loc), imag(BS_loc), 'r^', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Macro BS');
% 
% % 绘制用户位置并使用不同颜色
% for k = 1:length(BS_loc)
%     users_in_bs = MS_loc(user_assignment == k); % 找到分配给该基站的用户
%     scatter(real(users_in_bs), imag(users_in_bs), 50, colors(k, :), 'filled', 'DisplayName', ['Users of BS ' num2str(k)]);
% end
% 
% % 设置坐标范围和标签
% axis equal;
% box on;
% axis([-1.3 1.3 -1.3 1.3]); % 设置显示范围
% xlabel('x axis position (km)', 'FontName', 'Times New Roman');
% ylabel('y axis position (km)', 'FontName', 'Times New Roman');
% % title('7-cellHexagonal Cell Deployment with User Assignment');
% 
% % 添加图例和网格
% % legend('Macro BS', 'Location', 'bestoutside');
% % grid on;
% hold off;


%% compute MS-BS distance
dist = NaN(K,L);  % MS-BS distance
BS_loc_virtual = BS_loc; 
for u = 1:num_MS_per_cell
    dist(u,:) = abs(BS_loc_virtual - MS_loc(u));
end

for m = 2:L
    BS_loc_virtual = BS_loc;
    
    % v = mod(m,6) + 2; w = m; % map v to w
    % BS_loc_virtual(v) = BS_loc(m) + BS_loc(w); 
    % 
    % v = mod(m+1,6) + 2; w = mod(m-3,6) + 2;
    % BS_loc_virtual(v) = BS_loc(m) + BS_loc(w); 
    % 
    % v = mod(m+2,6) + 2; w = mod(m-7,6) + 2;
    % BS_loc_virtual(v) = BS_loc(m) + BS_loc(w);
    
    for u = (1:num_MS_per_cell) + (m-1)*num_MS_per_cell
        dist(u,:) = abs(BS_loc_virtual - MS_loc(u));
    end
end

% chn fading
dist = max(dist, 5e-3);
pathLoss = 128.1 + 37.6*log10(dist) + stand*randn([K,L]);
distPathLoss = 10.^(-pathLoss/10);

% chn coefficient
M = mimo_pattern(1); N = mimo_pattern(2);
tau = [0 200 800 1200 2300 3700]*1e-9;
p_db = [0 -0.9 -4.9 -8.0 -7.8 -23.9];
p = 10.^(p_db/10);
num_path = length(p);
p = p/sum(p);
ampli = sqrt(p);
% T = 8; % num of subcarriers
df = BW*1e6/T;
fc = (1:T)*df;
phase = exp(-1i*2*pi*tau'*fc);

chn = nan(N,M,T,K,L);
for i = 1:K
    for j = 1:L 
        for a = 1:M
            for b = 1:N
                if T==1
                    x = randn() + 1i*randn();
                    chn(b,a,:,i,j) = sqrt(distPathLoss(i,j))*x/norm(x);
                else
                    m = 1/sqrt(2)*( randn(1,num_path) + 1i*randn(1,num_path) );
                    m = m.*ampli;     
                    fading_en = sum(diag(m)*phase);
                    chn(b,a,:,i,j) = sqrt(distPathLoss(i,j))*fading_en;
                end
            end
        end
    end
end


% chn = ones(N,M,T,K,L)*1e-30;
% for i = 1:K
%     for j = 1:L 
%         for a = 1:1
%             for b = 1:1
%                 m = 1/sqrt(2)*( randn(1,num_path) + 1i*randn(1,num_path) );
%                 m = m.*ampli;     
%                 fading_en = sum(diag(m)*phase);
%                 chn(b,a,:,i,j) = sqrt(chnMagnitude(i,j))*fading_en;
%             end
%         end
%     end
% end

end