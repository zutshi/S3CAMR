function dump_model()
dump_xt();
% dump_x();
end

function dump_x()
SYS_NAME = 'vdp_x_1e6';
IP_FILE = ['./' SYS_NAME '_data.mat'];
MODEL_FILE = ['./' SYS_NAME '_flat_model'];
ALL_FILE = ['./' SYS_NAME '_all'];
model_delta_t = 0.01;
DATA = load(IP_FILE);

X = DATA.Y_summary(:, 1:2);
Y = DATA.Y_summary(:, 3:4);
% Create abstraction
eps = [1,1];
%Range = [min(X)', max(X)'];
Range = [-1.9999, 1.9999; -7.9999 7.9999];
fprintf('getting model...\n')
model = Model(Range, eps, Y, X, 'x');

GA = GridAbstraction(eps);
% RA = RelAbs(model, GA, Range, model_delta_t);
RA = RelAbs(model, model_delta_t, eps);
fm = model.flat();
save(MODEL_FILE, 'fm');
save(ALL_FILE);
end



function dump_xt()
SYS_NAME = 'vdp_xt_1e6' ;
IP_FILE = ['./' SYS_NAME '_data.mat'];
MODEL_FILE = ['./' SYS_NAME '_flat_model'];
ALL_FILE = ['./' SYS_NAME '_all'];

DATA = load(IP_FILE);

X = DATA.Y_summary(:, 1:3);
Y = DATA.Y_summary(:, 4:5);
% Create abstraction
eps = [0.5,0.5,0.01];
%Range = [min(X)', max(X)'];
Range = [-1.9999, 1.9999; -7.9999 7.9999; 0 0.01];
fprintf('getting model...\n')
model = Model(Range, eps, Y, X, 'xt');
model_delta_t = NaN;
% GA = GridAbstraction(eps);
RA = RelAbs(model, model_delta_t, eps);
fm = model.flat();

fprintf('writing files: %s, %s\n', MODEL_FILE, ALL_FILE);
save(MODEL_FILE, 'fm');
save(ALL_FILE);
fprintf('done!\n');
end
