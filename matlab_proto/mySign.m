% Matlab's sign() doesn't handle sign(0) properly! returns 0!!
% X_snapped = sign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat + grid_eps_mat/2);
% this one returns sign(0) = 1
function s = mySign(X)
s = abs(X)./X;
nanIdx = isnan(s);
% replace sign(0) with 1
s(nanIdx) = 1;
end