
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
model = Model(Range, eps, [Y1, Y2], X);

GA = GridAbstraction(eps);
RA = RelAbs(model, GA, Range, model_delta_t);
fm = RA.get_flattened_model();
save('flat_model', 'fm')
