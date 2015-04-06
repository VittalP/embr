% Set up global paths used throughout the code
%addpath learning;
%addpath detection;
%addpath visualization;
addpath evaluation;
addpath third_party_code;
addpath(genpath('~/export_fig/'));

if isunix()
%  addpath mex_unix;
elseif ispc()
  addpath mex_pc;
end

% Path to DivMBest
DivMBest_PATH = '~/divmbest/';
DivMBest_pose_estimation_PATH = [DivMBest_PATH 'pose_estimation/'];
%save('DivMBest_pose_estimation_PATH.mat', 'DivMBest_pose_estimation_PATH');
params.DivMBest_PATH = DivMBest_pose_estimation_PATH;

addpath(genpath(DivMBest_pose_estimation_PATH));

% directory for caching models, intermediate data, and results
cachedir = 'cache/';
if ~exist(cachedir,'dir')
  mkdir(cachedir);
end

if ~exist([cachedir 'imrotate/'],'dir')
  mkdir([cachedir 'imrotate/']);
end

if ~exist([cachedir 'imflip/'],'dir')
  mkdir([cachedir 'imflip/']);
end

parsedir = [DivMBest_pose_estimation_PATH './PARSE/'];
inriadir = './INRIA/';
