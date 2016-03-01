function dump_model(SYS_NAME)
% SYS_NAME = 'vdp_x_1e6';
% SYS_NAME = 'vdp_xt_1e7' ;
% SYS_NAME = 'vdp_xt_1e6' ;

ALL_FILE = FileNames.all_file_name(SYS_NAME);
MODEL_FILE = FileNames.model_file_name(SYS_NAME);
IP_FILE = FileNames.ip_file_name(SYS_NAME);

Data = load(IP_FILE);


% Create abstraction
% if model is xt
if isnan(Data.model_delta_t)
    model_type = 'xt';
%     eps = [1,1,0.01];
%     eps = [0.2,0.2,0.01];
%     Range = [-1.9999, 1.9999; -7.9999 7.9999; 0 0.00999];


    Range = [-1.9999, 1.9999; -7.9999 7.9999; 0 0.4999];
    eps = [0.5,0.2,0.05];
    
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
