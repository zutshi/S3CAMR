% N: num samples
% tol: x' = Ax +b +- tol
function simulate_model(N, tol)
fprintf('loading all associated data...\n')

% SYS_NAME = 'vdp_xt_1e6';
% model_type = 'xt';

SYS_NAME = 'vdp_x_1e6';
model_type = 'x';


ALL_FILE = ['./' SYS_NAME '_all'];
load(ALL_FILE);
fprintf('Done!\n')

x0 = [-0.4 0.4; -0.4 0.4];
prop = [-1, -0.7; -6.5 -5.6];
opts = struct('v', 0, 'p', 0);
fprintf('verifying paths...\n')
% my_figure(2)
% hold on
% plot_cell(x0, 'r', opts)
% plot_cell(prop, 'r', opts)

% X = genRandVectors(N, [-0.4 0.4; -0.4 0.4]);
X = genRandVectors(N, [0.4 0.4; -0.4 -0.4]);

if strcmp(model_type,'xt')
    simulate_and_test_model_xt(X,RA,tol);
elseif strcmp(model_type,'x')
    simulate_and_test_model_x(X,RA,tol);
else
    error('unkown model type!')
end
end

function simulate_and_test_model_x(X,RA,tol)
% figure(1);hold on
% figure(2);hold on
figure(3);hold on
T = 1;
% num discrete steps
n = T/RA.delta_t;

for i = 1:size(X, 1)
    x = X(i,:);
    
    [t, y_] = vdp_sim(x, [0 T]);
    figure(1);    plot(t, y_(:,1),'k-');
    figure(2);    plot(t, y_(:,2),'k-');
    figure(3);    plot(y_(:,1), y_(:,2),'k-');
    
    [t,y] = RA.simulate(x, n, 'x',tol);
    figure(1);    plot(t, y(:,1),'b-');
    figure(2);    plot(t, y(:,2),'b-');
    figure(3);    plot(y(:,1), y(:,2),'b-');
end

end

function simulate_and_test_model_xt(X,RA,tol)
figure(1);hold on
figure(2);hold on
figure(3);hold on

T = 1;
n = 100;
for i = 1:size(X, 1)
    x = X(i,:);
    [t, y_] = vdp_sim(x, [0 T]);
    %     plot(y_(:,1), y_(:,2), 'r*');
    figure(1);    plot(t(:,1), y_(:,1), 'k-');
    figure(2);    plot(t(:,1), y_(:,2), 'k-');
    figure(3);    plot(y_(:,1), y_(:,2), 'k-');
    
    [t,y] = RA.simulate(x,n, 'xt',tol);
    figure(1);    plot(t, y(:,1),'b-');
    figure(2);    plot(t, y(:,2),'b-');
    figure(3);    plot(y(:,1), y(:,2), 'b-');
    %     plot(y(:,1), y(:,2),'b*');
    %     [~,y] = RA.simulate(y(end,1:2),50, 'xt');
    %     plot(y(:,1), y(:,2),'b*');
end
% [~, y_] = vdp_sim(X, [0 T]);
% plot(y_(:,1), y_(:,2));
end