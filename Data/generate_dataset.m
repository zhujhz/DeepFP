function manifest = generate_dataset(varargin)
%GENERATE_DATASET One entry point for the DeepFP channel model.
% This uses the repository's GenerateNetwork7.m, as the original
% generate_save_chn_func.m and generate_deepfp_dataset.m did.
%
% Example (run from the repository's Data directory or with it on path):
%   generate_dataset('cells',3,'users_per_cell',6,'nt',64,'nr',4, ...
%       'train_samples',40000,'validation_samples',5000, ...
%       'test_samples',5000,'samples_per_file',1000, ...
%       'output_dir','Data/raw_three_cell_demo', ...
%       'filename_template','{split}_{part}_N{count}_BS{cells}.mat');
%
% A generated MAT file contains complex double `chn` in MATLAB order
% [sample, Nr, Nt, users_per_cell, receiving_cell, transmitting_cell].
% MAT files and generated_manifest.json remain in output_dir. Use
% `prepare_data.py generated` to write HDF5 to a separate Data/h5_* directory.

p = inputParser;
p.FunctionName = 'generate_dataset';
addParameter(p,'cells',1,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x == floor(x) && x >= 1 && x <= 7);
addParameter(p,'users_per_cell',6,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x == floor(x) && x > 0 && mod(x,3) == 0);
addParameter(p,'nt',64,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x == floor(x) && x > 0);
addParameter(p,'nr',4,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x == floor(x) && x > 0);
addParameter(p,'train_samples',1000,@nonnegative_integer);
addParameter(p,'validation_samples',100,@nonnegative_integer);
addParameter(p,'test_samples',100,@nonnegative_integer);
addParameter(p,'samples_per_file',1000,@(x) positive_integer(x));
addParameter(p,'seed',20261106,@(x) nonnegative_integer(x));
addParameter(p,'bandwidth_mhz',10,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x > 0);
addParameter(p,'spacing_km',0.8,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x > 0);
addParameter(p,'shadowing_std_db',8,@(x) isscalar(x) && isnumeric(x) && isfinite(x) && x >= 0);
addParameter(p,'output_dir',fullfile(fileparts(mfilename('fullpath')),'raw_generated'),@(x) ischar(x) || isstring(x));
addParameter(p,'filename_template','deepfp_{split}_{part}_N{count}_Nr{nr}_Nt{nt}_K{users}_BS{cells}_seed{seed}.mat',@(x) ischar(x) || isstring(x));
addParameter(p,'overwrite',false,@(x) islogical(x) && isscalar(x));
parse(p,varargin{:});
cfg = p.Results;

if cfg.train_samples + cfg.validation_samples + cfg.test_samples == 0
    error('generate_dataset:Empty','At least one split must contain samples.');
end
data_dir = char(cfg.output_dir);
template = char(cfg.filename_template);
if isempty(strfind(template,'{split}')) || isempty(strfind(template,'{part}')) %#ok<STREMP>
    error('generate_dataset:Template','filename_template must include {split} and {part}.');
end
if ~endsWith(template,'.mat')
    error('generate_dataset:Template','filename_template must end in .mat.');
end

split_names = {'train','validation','test'};
split_counts = [cfg.train_samples,cfg.validation_samples,cfg.test_samples];
files = struct('split',{},'part',{},'samples',{},'name',{});
for s = 1:numel(split_names)
    for part = 0:ceil(split_counts(s)/cfg.samples_per_file)-1
        count = min(cfg.samples_per_file,split_counts(s)-part*cfg.samples_per_file);
        name = make_filename(template,split_names{s},part,count,cfg);
        [folder,~,ext] = fileparts(name);
        if ~isempty(folder) || ~strcmpi(ext,'.mat') || strcmp(name,'.') || strcmp(name,'..')
            error('generate_dataset:Filename','Template must produce a plain .mat filename.');
        end
        files(end+1) = struct('split',split_names{s},'part',part, ...
            'samples',count,'name',name); %#ok<AGROW>
    end
end
names = {files.name};
if numel(unique(names)) ~= numel(names)
    error('generate_dataset:Filename','Template produced duplicate filenames.');
end
manifest_path = fullfile(data_dir,'generated_manifest.json');
if ~cfg.overwrite && exist(manifest_path,'file')
    error('generate_dataset:Exists','Manifest already exists: %s',manifest_path);
end
for f = 1:numel(files)
    dest = fullfile(data_dir,files(f).name);
    if ~cfg.overwrite && exist(dest,'file')
        error('generate_dataset:Exists','Output already exists: %s',dest);
    end
end
if ~exist(data_dir,'dir'), mkdir(data_dir); end

generator_dir = fileparts(mfilename('fullpath'));
addpath(generator_dir);
for s = 1:numel(split_names)
    rng(cfg.seed + s - 1,'twister');
    selected = find(strcmp({files.split},split_names{s}));
    for index = selected
        count = files(index).samples;
        chn = complex(zeros(count,cfg.nr,cfg.nt,cfg.users_per_cell,cfg.cells,cfg.cells));
        for n = 1:count
            raw = GenerateNetwork7(cfg.bandwidth_mhz,cfg.cells, ...
                cfg.cells*cfg.users_per_cell,[cfg.nt,cfg.nr],1, ...
                cfg.spacing_km,cfg.shadowing_std_db);
            % raw: [Nr,Nt,1,all_users,transmitting_cell]. MATLAB's
            % column-major reshape groups users by receiving cell.
            chn(n,:,:,:,:,:) = reshape(raw,[1,cfg.nr,cfg.nt, ...
                cfg.users_per_cell,cfg.cells,cfg.cells]);
            if mod(n,100) == 0 || n == count
                fprintf('%s part %d: %d/%d\n',split_names{s},files(index).part,n,count);
            end
        end
        split = split_names{s};
        part = files(index).part;
        destination = fullfile(data_dir,files(index).name);
        save(destination,'chn','cfg','split','part','-v7.3');
        fprintf('Saved %s\n',destination);
    end
end

manifest = struct('format_version',1,'generator','GenerateNetwork7', ...
    'shape_convention','sample,Nr,Nt,user,receiving_cell,transmitting_cell', ...
    'config',cfg,'files',files);
fid = fopen(manifest_path,'w');
if fid < 0, error('generate_dataset:Manifest','Cannot write %s',manifest_path); end
try
    fprintf(fid,'%s',jsonencode(manifest));
catch exception
    fclose(fid);
    rethrow(exception);
end
fclose(fid);
fprintf('Saved %s\n',manifest_path);
end

function yes = nonnegative_integer(x)
yes = isscalar(x) && isnumeric(x) && isfinite(x) && x == floor(x) && x >= 0;
end

function yes = positive_integer(x)
yes = nonnegative_integer(x) && x > 0;
end

function name = make_filename(template,split,part,count,cfg)
tokens = {'{split}','{part}','{count}','{total}','{nr}','{nt}', ...
    '{users}','{cells}','{seed}'};
totals = struct('train',cfg.train_samples,'validation',cfg.validation_samples, ...
    'test',cfg.test_samples);
values = {split,sprintf('%03d',part),num2str(count),num2str(totals.(split)), ...
    num2str(cfg.nr),num2str(cfg.nt),num2str(cfg.users_per_cell), ...
    num2str(cfg.cells),num2str(cfg.seed)};
name = template;
for i = 1:numel(tokens)
    name = strrep(name,tokens{i},values{i});
end
if ~isempty(regexp(name,'[{}]','once'))
    error('generate_dataset:Template','Unknown filename token in %s',name);
end
end
