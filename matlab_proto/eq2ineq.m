%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Works only for 2 dimensions!!!!
% Hard to generelize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Modify Equality constraints to inequality as follows
% Ax = b => Ax <= b /\ -Ax <= -b
function [A,b] = eq2ineq(Aeq, beq, nd, eps)
r = length(beq);
% r = size(Aeq,1);
% create an indexing scheme such that ever block of row is repeated twice
x = [0:(nd-1) -(nd-1):0];
% Hard to generate the indexing for arbitary dimensions
idx = repelem(1:r,1,nd) + repmat(x,1,r/2);

% Negate the alternate rows

A = Aeq(idx,:);
b = beq(idx,:);

A = A .* repmat(repelem([1;-1],nd),size(A,1)/(2*nd),size(A,2));
b = b .* repmat(repelem([1;-1],nd),size(b,1)/(2*nd),1);
%     relax the constraints now
b = b + eps;
end