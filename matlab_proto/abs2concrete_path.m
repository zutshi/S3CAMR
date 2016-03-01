% Expects cell sequences and ranges for x0 and prop
function [Y,e] = abs2concrete_path(RA, abs_path, x0, prop, dyn_cons_type, tol)
% abs_path
% dynamics should be encoded as 'inequality' or 'equality'?
EQ = 'equality';
INEQ = 'inequality';
Y = [];

%% num_dims
nd = size(x0, 1);
l = size(abs_path, 1) + 2;

[AA,bb,LB,UB,cA,B] = get_constraints(RA, abs_path, x0, prop, l, nd);
f = zeros(l*nd, 1);

% relax the constraints a bit
% LB = LB - 0.05;
% UB = UB + 0.05;
if strcmp(dyn_cons_type, EQ)
    A = []; b = [];
    Aeq = AA; beq = bb;
    e = 0;
elseif strcmp(dyn_cons_type, INEQ)
    Aeq = []; beq = [];
    % Modify Equality constraints to inequality as follows
    % Ax = b => Ax <= b /\ -Ax <= -b
    [A,b] = eq2ineq(AA, bb, nd, tol);
else
    error('unknown option %s\n', dyn_cons_type);
end

options = optimoptions('linprog','Algorithm','dual-simplex');
[X,fval,exitflag] = linprog(f,A,b,Aeq,beq,LB,UB,[],options);
if exitflag == 1
    fprintf('path found in RA\n');
    Y = reshape(X,nd,length(X)/nd)';
    e = compute_used_tol(Y,cA,B);
elseif exitflag == -2
    fprintf('path infeasible in RA\n');
else
    fprintf('error code returned: %d\n', exitflag);
end
end

function e = compute_used_tol(Y,cA,B)
% num dim
nd = size(Y,2);

% trace length
n = size(Y,1)-1;

% reshape B to make computations easy
B = reshape(B,nd,size(B,1)/nd)';

e = zeros(n,nd);
for i = 1:n
    x_ = cA{i}*Y(i,:)' + B(i,:)';
    e(i,:) = Y(i+1,:) - x_';
end

% print the dynamics used
% cell2mat(cA)'
% B
end

% returns cA and B for debug info
function [Aeq,beq,LB,UB,cA,B] = get_constraints(RA, abs_path, x0, prop, l, nd)
%% Problem setup
% [A0nxn  -Inxn  0nxn   0nxn] [X0]    [b0]
% [ 0nxn  A1nxn  -Inxn  0nxn] [X1] = -[b1]
% [ 0nxn  0nxn   A2nxn -Inxn] [X2]    [b2]
%                             [X3]

%% init
A = [];
b = [];

%% Add the initial state
LB = x0(:,1);
UB = x0(:,2);

[~,c0] = Grid.generateCellsFromRange(x0, RA.eps);
assert(size(c0,1)==1);
sm0 = RA.get_cell_model(c0);
dyn0 = sm0.M;
cA = {dyn0.A};
B = [dyn0.b];

%% Add the rest of the cells
for i = 1:l-2
    c = abs_path(i,:);
    sub_model = RA.get_cell_model(c);
    dyn = sub_model.M;
    cA = [cA dyn.A];
    B = [B; dyn.b];
    crange = Grid.getCellRange(c, RA.eps);
    LB = [LB; crange(:,1)];
    UB = [UB; crange(:,2)];
end

%% Add the property region
LB = [LB; prop(:,1)];
UB = [UB; prop(:,2)];

bdA = blkdiag(cA{:});
bdA = padarray(bdA,[0 nd],0,'post');

% create a right shifted identity matrix
% I = padarray(padarray(eye(nd*(l-1)),[0 nd],0,'pre'), [nd 0],0,'post');
I = padarray(eye(nd*(l-1)),[0 nd],0,'pre');

%% setup a linprog
Aeq = bdA - I;
beq = -B;
end

% % Modify Equality constraints to inequality as follows
% % Ax = b => Ax <= b /\ -Ax <= -b
% function [A,b] = eq2ineq(Aeq, beq, nd, eps)
% r = length(beq);
% % create an indexing scheme such that ever block of row is repeated twice
% % idx = repelem(1:r/nd,1,nd) + repmat([0 1 -1 0],1,5);
% x = [0:nd-1 nd-1:0];
% idx = repelem(1:r/nd,1,nd) + repmat(x,1,r/length(x));
% 
% % Negate the alternate rows
% A = Aeq(idx,:);
% % b = beq(idx,:);
% A = A .* repmat([1;-1],size(A,1)/2,nd);
% % b = b .* repmat([1;-1],size(b,1)/2,1);
% %     relax the constraints now
% % b = b + eps;
% b = [];
% end
