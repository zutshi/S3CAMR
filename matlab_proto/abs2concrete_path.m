% Expects cell sequences and ranges for x0 and prop
function Y = abs2concrete_path(RA, abs_path, x0, prop, dyn_cons_type, eps)

% dynamics should be encoded as 'inequality' or 'equality'?
EQ = 'equality';
INEQ = 'inequality';
Y = [];

%% num_dims
nd = size(x0, 1);
l = size(abs_path, 1) + 2;

[AA,bb,LB,UB] = get_constraints(RA, abs_path, x0, prop, l, nd);
f = zeros(l*nd, 1);

% relax the constraints a bit
% LB = LB - 0.05;
% UB = UB + 0.05;
if strcmp(dyn_cons_type, EQ)
    A = []; b = [];
    Aeq = AA; beq = bb;
elseif strcmp(dyn_cons_type, INEQ)
    Aeq = []; beq = [];
    % Modify Equality constraints to inequality as follows
    % Ax = b => Ax <= b /\ -Ax <= -b
    [A,b] = eq2ineq(AA, bb, nd, eps);
else
    error('unknown option %s\n', dyn_cons_type);
end

options = optimoptions('linprog','Algorithm','dual-simplex');
[X,fval,exitflag] = linprog(f,A,b,Aeq,beq,LB,UB,[],options);
if exitflag == 1
    fprintf('path found in RA\n');
    Y = reshape(X,nd,length(X)/nd)';
elseif exitflag == -2
    fprintf('path infeasible in RA\n');
else
    fprintf('error code returned: %d\n', exitflag);
end
end

function [Aeq,beq,LB,UB] = get_constraints(RA, abs_path, x0, prop, l, nd)
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

[~,c0] = RA.GA.generateCellsFromRange(x0);
assert(size(c0,1)==1);
dyn0 = RA.get_cell_dyn(c0);
cA = {dyn0.A};
B = [dyn0.b];

%% Add the rest of the cells
for i = 1:l-2
    c = abs_path(i,:);
    dyn = RA.get_cell_dyn(c);
    cA = [cA dyn.A];
    B = [B; dyn.b];
    crange = RA.GA.getCellRange(c);
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
