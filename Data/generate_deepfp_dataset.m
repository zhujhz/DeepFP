function generate_deepfp_dataset(nTrain, nTest, seed)
% Generate independent seven-cell channel sets in the HDF5 MAT format.
% Run from MATLAB after adding this file and GenerateNetwork7.m to the path.
if nargin < 1, nTrain = 500; end
if nargin < 2, nTest = 50; end
if nargin < 3, seed = 20261104; end

source_dir = fileparts(which('GenerateNetwork7'));
assert(~isempty(source_dir), 'GenerateNetwork7.m must be on the MATLAB path.');
data_dir = fullfile(source_dir, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

cfg = struct('N_t', 64, 'N_r', 4, 'users_per_cell', 6, ...
    'cells', 7, 'bandwidth_MHz', 10, 'spacing_km', 0.8, ...
    'shadowing_std_dB', 8, 'seed', seed);
rng(seed, 'twister');

generate_split('train', nTrain, cfg, data_dir);
generate_split('test', nTest, cfg, data_dir);
end

function generate_split(split_name, nSamples, cfg, data_dir)
N_t = cfg.N_t;
N_r = cfg.N_r;
K = cfg.users_per_cell;
B = cfg.cells;
chn = complex(zeros(nSamples, N_r, N_t, K, B, B));
for i = 1:nSamples
    raw = GenerateNetwork7(cfg.bandwidth_MHz, B, B*K, ...
        [N_t, N_r], 1, cfg.spacing_km, cfg.shadowing_std_dB);
    % raw: [N_r,N_t,1,B*K,B]. Users are grouped by receiving cell.
    chn(i, :, :, :, :, :) = reshape(raw, [1, N_r, N_t, K, B, B]);
    if mod(i, 25) == 0 || i == nSamples
        fprintf('%s %d/%d\n', split_name, i, nSamples);
    end
end

filename = sprintf('%s_%d_Nr_%d_Nt_%d_N_user_%d_BS_%d.mat', ...
    split_name, nSamples, N_r, N_t, K, B);
save(fullfile(data_dir, filename), 'chn', 'cfg', '-v7.3');
fprintf('Saved %s\n', fullfile(data_dir, filename));
end
