% % N = 100;
% N = 70; % lets make it easier
% % Stop after 5 abstract paths have been found
% thresh = 4;
% tol is inequality tolerence
function falsify(N, thresh, tol, opts)

% x0 = [-0.4 0.4; -0.4 0.4];
x0 = [0.3 0.4; -0.4 -0.3];
prop = [-1, -0.7; -6.5 -5.6];

FILE = './vdp_data_1e6.mat';
model_delta_t = 0.01;
DATA = load(FILE);

X = DATA.Y_summary(:, 1:2);
Y1 = DATA.Y_summary(:, 3);
Y2 = DATA.Y_summary(:, 4);
% Create abstraction
eps = [1,1];
%Range = [min(X)', max(X)'];
Range = [-2, 2; -8 8];
fprintf('getting model...\n')
model = get_vdp_model(Range, eps, Y1, Y2, X);

GA = GridAbstraction(eps);
RA = RelAbs(model, GA, Range, model_delta_t);

% RA.verify_model(X, Y1, Y2);
% return

% simulate_and_test_model(RA);

fprintf('verifying paths...\n')
my_figure(2)
hold on
plot_cell(prop, 'r', opts)
dyn_cons_type = 'inequality';
if thresh == inf
    get_paths_gen(RA, N, x0, prop, dyn_cons_type, model_delta_t, tol, opts);
else
    get_paths(RA, N, x0, prop, dyn_cons_type, model_delta_t, tol, thresh, opts);
end

end

function get_paths_gen(RA, N, x0, prop, dyn_cons_type, model_delta_t, tol, opts)
fprintf('enumerating abstract paths of length...\n')
path_gn = find_all_paths_gen(RA, N, x0, prop, opts);

fprintf('Entering an infinite loop...press ctrl+c to cancel anytime. Press any key to acknowlege.')
pause();
while 1
    hold on
    p = path_gn();
    Y = abs2concrete_path(RA, p, x0, prop, dyn_cons_type, tol);
    if isempty(Y)
        continue
    end
    plot(Y(:,1),Y(:,2), '.');
    X0 = Y(1,:);
    % Compute Time horizon
    T = (size(Y,1)-1) * model_delta_t;
    [~,Y_] = vdp_sim(X0, [0 T]);
    plot(Y_(:,1),Y_(:,2), '*');
    drawnow
    pause()
end
end

function get_paths(RA, N, x0, prop, dyn_cons_type, model_delta_t, tol, thresh, opts)
paths = find_all_paths_thresh(RA, N, x0, prop, thresh, opts);
for i = 1:length(paths)
    hold on
    p = paths{i};
    Y = abs2concrete_path(RA, p, x0, prop, dyn_cons_type, tol);
    if isempty(Y)
        continue
    end
    plot(Y(:,1),Y(:,2), '.');
    X0 = Y(1,:);
    % Compute Time horizon
    T = (size(Y,1)-1) * model_delta_t;
    [~,Y_] = vdp_sim(X0, [0 T]);
    plot(Y_(:,1),Y_(:,2), '*');
    drawnow
end
end

function simulate_and_test_model(RA)
my_figure(1)
hold on
X = genRandVectors(100, [-0.4 0.4; -0.4 0.4]);
for i = 1:size(X, 1)
    x = X(i,:);
    [~,y] = RA.simulate(x, 100);
    plot(y(:,1), y(:,2));
end
end
