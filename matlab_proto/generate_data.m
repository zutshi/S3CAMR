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
vdp_xt_data()
%vdp_x_data()
end

function vdp_xt_data()
filename = 'vdp_xt_1e7_data';
N = 1e7;
NUM_STATE_VARS = 2; % x1, x2, t
NUM_INDP_VARS = NUM_STATE_VARS + 1; % num(independant vectors) = num_states + time 
NUM_DEP_VARS = NUM_STATE_VARS

Y_summary = zeros(N, NUM_INDP_VARS+NUM_DEP_VARS);
Tspan = [0 0.5]; % more like a delta_t span, as the system is time independant
X0_set = [-2 2; -8 8];
I0_set = [X0_set; Tspan]
I0_samples = genRandVectors(rand(N, NUM_INDP_VARS), I0_set);
optionsODE = odeset('Refine',1,'RelTol',1e-6);
tidx = NUM_INDP_VARS


parfor i = 1:N
    I0 = I0_samples(i, :);

    % Ignores all points on the traces apart from the end. Wastage?
    % How can we use them?
    [~,Y] = ode45(@vdp_dyn,[0 I0_samples(i, tidx)],I0(1:end-1),optionsODE);   

    % Y_summary layout:
    % [indp1(x0) indp2(x1) indp3(t) dep1(y0=x0') dep2(y1=x1')]
    Y_summary(i, :) = [Y(1, :) I0_samples(i, tidx) Y(end, :)];
end

save(filename, 'Y_summary');
end

function vdp_x_data()
% add_HyAutSim_to_path();
% [sys_def,sys_prop,sys_opt] = vanderpol();
% sim_fn = hybrid_system_simulator(sys_def,sys_prop,sys_opt);

NUM_STATE_VARS = 2;
N = 1e7;

filename = 'vdp_data_1e7';
% vdp_data = matfile(filename,'Writable',true);

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
% my_figure(1)
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
