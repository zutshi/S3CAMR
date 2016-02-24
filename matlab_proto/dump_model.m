function dump_model()
dump_xt();
%dump_x();
end

function dump_x()
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
end



function dump_xt()

FILE = './vdp_xt_1e6_data.mat';
DATA = load(FILE);

X = DATA.Y_summary(:, 1:3);
Y = DATA.Y_summary(:, 3:4);
% Create abstraction
eps = [1,1,0.5];
%Range = [min(X)', max(X)'];
Range = [-2, 2; -8 8; 0 1];
fprintf('getting model...\n')
model = Model(Range, eps, Y, X);

GA = GridAbstraction(eps);
RA = RelAbs(model, GA, Range, 0);
fm = RA.get_flattened_model();
save([FILE 'flat_model_'], 'fm')
end
