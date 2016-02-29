function dump_model()
% SYS_NAME = 'vdp_x_1e6';
SYS_NAME = 'vdp_xt_1e6' ;

% model_delta_t = 0.01;
% X = DATA.Y_summary(:, 1:2);
% Y = DATA.Y_summary(:, 3:4);
%
% X = DATA.Y_summary(:, 1:3);
% Y = DATA.Y_summary(:, 4:5);
% % model_type
% model_delta_t
% Range

IP_FILE = ['./' SYS_NAME '_data.mat'];
MODEL_FILE = ['./' SYS_NAME '_flat_model'];
ALL_FILE = ['./' SYS_NAME '_all'];
Data = load(IP_FILE);


% Create abstraction
% if model is xt
if isnan(Data.model_delta_t)
    model_type = 'xt';
    eps = [1,1,0.01];
    Range = [-1.9999, 1.9999; -7.9999 7.9999; 0 0.00999];
else
    model_type = 'x';
    eps = [1,1];
    Range = [-1.9999, 1.9999; -7.9999 7.9999];
end

fprintf('getting model...\n')
model = Model(Range, eps, Data.Y, Data.X, model_type);
RA = RelAbs(model, Data.model_delta_t, eps);
fm = model.flat();

fprintf('writing files: %s, %s\n', MODEL_FILE, ALL_FILE);
save(MODEL_FILE, 'fm');
save(ALL_FILE);
fprintf('done!\n');

end

function fm = dump_x()
% Create abstraction
eps = [1,1];
%Range = [min(X)', max(X)'];
% Range = [-1.9999, 1.9999; -7.9999 7.9999];
fprintf('getting model...\n')
model = Model(Range, eps, Y, X, model_type);

GA = GridAbstraction(eps);
% RA = RelAbs(model, GA, Range, model_delta_t);
RA = RelAbs(model, model_delta_t, eps);
fm = model.flat();
end



function fm = dump_xt()
% Create abstraction
eps = [0.5,0.5,0.01];
%Range = [min(X)', max(X)'];

% Range = [-1.9999, 1.9999; -7.9999 7.9999; 0 0.01];

fprintf('getting model...\n')
model = Model(Range, eps, Y, X, model_type);
% model_delta_t = NaN;
% GA = GridAbstraction(eps);
RA = RelAbs(model, model_delta_t, eps);
fm = model.flat();
end
