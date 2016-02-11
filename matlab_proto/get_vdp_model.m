function model = get_vdp_model(Range, eps, Y1, Y2, X)

[num_points, num_dims] = size(X);

nr = (Range(1,2) - Range(1,1))/eps(1);
nc = (Range(2,2) - Range(2,1))/eps(2);
num_paritions = nr*nc;
model = cell(nr, nc);

% TODO: Replace by NDGRID
i_ = 1;
j_ = 1;
for i = Range(1,1):eps(1):Range(1,2)-eps(1)
    for j = Range(2,1):eps(2):Range(2,2)-eps(2)
        idx = find(X(:,1) >= i & X(:,1) <= i+eps(1) & X(:,2) >= j & X(:,2) <= j+eps(2));
        y1 = Y1(idx);
        y2 = Y2(idx);
        x = X(idx, :);
        % append a column of ones to make the regress function compute
        % b in Ax + b
        x_ones = [x ones(size(x,1),1)];
        b1 = regress(y1, x_ones);
        b2 = regress(y2, x_ones);
%         A = [b1(1:end-1,1) b2(1:end-1,1)];
%         b = [b1(end); b2(end)];
        A = [b1(1:end-1) b2(1:end-1)]';
        b = [b1(end); b2(end)];
        dyn = struct('A', A, 'b', b, 'idx', idx);
        model{i_, j_} = dyn;
        j_ = j_ + 1;
    end
    i_ = i_ + 1;
    j_ = 1;
end
end
