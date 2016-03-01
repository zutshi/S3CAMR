% % N = 100;
% N = 70; % lets make it easier
% % Stop after 5 abstract paths have been found
% thresh = 4;
% tol is inequality tolerence
% opts.v = verbosity
% opts.p = plot
function falsify(N, thresh, tol, opts)

x0 = [-0.4 0.4; -0.4 0.4];
% x0 = [0.3 0.4; -0.4 -0.3];
prop = [-1, -0.7; -6.5 -5.6];

% FILE = './vdp_x_1e6_data.mat';
FILE = './vdp_xt_1e6_data.mat';

% model_delta_t = 0.01;
Data = load(FILE);

% Range = [-2, 2; -8 8];

% Create abstraction
% eps = [1,1];
% Range = [-1.9999, 1.9999; -7.9999 7.9999];

eps = [1,1,0.01];
Range = [-1.9999, 1.9999; -7.9999 7.9999; 0 0.00999];

fprintf('getting model...\n')
model = Model(Range, eps, Data.Y, Data.X, '');

RA = RelAbs(model, Data.model_delta_t, eps);

% RA.verify_model(X, Y1, Y2);
% return


fprintf('verifying paths...\n')
my_figure(2)
hold on
plot_cell(prop, 'r', opts)


dyn_cons_type = 'inequality';
if thresh == inf
    get_paths_gen(RA, N, x0, prop, dyn_cons_type, Data.model_delta_t, tol, opts);
else
    get_paths(RA, N, x0, prop, dyn_cons_type, Data.model_delta_t, tol, thresh, opts);
end

end

function get_paths_gen(RA, N, x0, prop, dyn_cons_type, model_delta_t, tol, opts)
fprintf('enumerating abstract paths of length...\n')
path_gn = find_all_paths_gen(RA, N, x0, prop, opts);

fprintf('Entering an infinite loop...press ctrl+c to cancel anytime. Press any key to acknowlege.')
pause();
while 1
    p = path_gn();
    [Y,e] = abs2concrete_path(RA, p, x0, prop, dyn_cons_type, tol);
    if isempty(Y)
        continue
    end
    plot(Y(:,1),Y(:,2), '.');
    
    X0 = Y(1,:);
    
    % compare with pwa model
    [~,Y_] = RA.simulate(X0, size(Y,1)-1, 'x',tol);
    plot(Y_(:,1),Y_(:,2), '.');
    
    % Compute Time horizon
    T = (size(Y,1)-1) * model_delta_t;
    % compare with original model: sim()
    [~,Y__] = vdp_sim(X0, [0 T]);
    plot(Y__(:,1),Y__(:,2), '*');
    drawnow
    pause()
end
end

function get_paths(RA, N, x0, prop, dyn_cons_type, model_delta_t, tol, thresh, opts)

paths = find_all_paths_thresh(RA, N, x0, prop, thresh, opts);
% paths = find_all_cell_paths_thresh(RA, N, x0, prop, thresh, opts);

for i = 1:length(paths)
    p = paths{i};
    [Y,e] = abs2concrete_path(RA, p, x0, prop, dyn_cons_type, tol);
    if isempty(Y)
        continue
    end
    plot(Y(:,1),Y(:,2), 'r-*');
    X0 = Y(1,:);
    
    % compare with pwa model
    [~,Y_] = RA.simulate(X0, size(Y,1)-1, 'x',e);
    plot(Y_(:,1),Y_(:,2), 'k-*');
    
    % Compute Time horizon
    T = (size(Y,1)-1) * model_delta_t;
    % compare with original model: sim()
    [~,Y__] = vdp_sim(X0, [0 T]);
    plot(Y__(:,1),Y__(:,2), 'b-*');
    drawnow
end
end