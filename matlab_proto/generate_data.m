%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Aditya Zutshi
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% N = 1e7, takes 4848 seconds and ~300MB
%

function generate_data()
% add_HyAutSim_to_path();
% [sys_def,sys_prop,sys_opt] = vanderpol();
% sim_fn = hybrid_system_simulator(sys_def,sys_prop,sys_opt);

filename = 'vdp_data';
% vdp_data = matfile(filename,'Writable',true);

NUM_STATE_VARS = 2;
N = 1e7;

Y_summary = zeros(N, 2*NUM_STATE_VARS);
Tspan = [0 0.01];

X0_set = [-2 2; -8 8];
% X0_set = [-0.4 0.4; -0.4 0.4];
X0_samples = genRandVectors(rand(N, NUM_STATE_VARS), X0_set);
optionsODE = odeset('Refine',1,'RelTol',1e-6);

% parfor_progress(N);
parfor i = 1:N
    X0 = X0_samples(i, :);
    [~,Y] = ode45(@vdp_dyn,Tspan,X0,optionsODE);   
    Y_summary(i, :) = [Y(1, :) Y(end, :)];
%     parfor_progress;
end
% parfor_progress(0);

% vdp_data.Y_summary = Y_summary;
save(filename, 'Y_summary');

%% plot
% figure(1)
% hold on;
% Y_start = Y_summary(:, 1:NUM_STATE_VARS);
% Y_end = Y_summary(:, NUM_STATE_VARS+1:2*NUM_STATE_VARS);
% for i = 1:N
%     YY = [Y_start(i, :); Y_end(i, :)];
%     plot(YY(:, 1), YY(:, 2));
% end
end


function Y = vdp_dyn(~,X)
Y(1) = X(2);
Y(2) = 5 * (1 - X(1)^2) * X(2) - X(1);
Y = Y';
end


function add_HyAutSim_to_path()
addpath /home/zutshi/work/RA/cpsVerification/HyCU/releases/HyAutSim/
addpath /home/zutshi/work/RA/cpsVerification/HyCU/releases/HyAutSim/examples/
end
