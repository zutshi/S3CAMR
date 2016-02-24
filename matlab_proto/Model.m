classdef Model

    properties
        model;
        old_model;
        offset;
        % len_array
        la;
    end
    methods
        function obj = Model(range, eps, Y, X)
        obj.offset = -(min(range,[],2)' + eps/2) + 1;


        ndi = size(range,1);
        len_array = zeros(1, ndi);
        for i = 1:ndi
            len_array(i) = (range(i,2) - range(i,1))/eps(i);
        end
        obj.la = len_array

        old_model = Model_old(range, eps, Y(:,1), Y(:,2), X);
        % New function is 3 times slower!
        new_model = obj.Model_new(range, eps, Y, X);
        obj.old_model = old_model;

        obj.model = new_model;
        compare_models(old_model, obj);
        end
        function model = Model_new(obj, range, eps, Y, X)
            % num_samples, input dim
            [ns, ndi] = size(X);
            % num_samples_Y, output dim
            [nsY, ndo] = size(Y);
            assert(ns == nsY);


            % num of elements in each dim
            ne = diff(range, 1, 2)./eps';
            % num_paritions
            np = prod(ne);
            fprintf('number of paritions: %d\n', np);
            % create a flat array
            model = cell(1,np);

            sub_cells = generateCellsFromRange(range, eps);

            for i = 1:size(sub_cells,1)
                sbcl = sub_cells(i,:);
                LB = sbcl - eps/2;
                UB = sbcl + eps/2;

                idx = find(all(X >= repmat(LB,ns,1) & X <= repmat(UB,ns,1), 2));

                xi = X(idx, :);
                yi = Y(idx, :);
                nidx = length(idx);
                % append a column of ones to make the regress function compute
                % b in Ax + b
                x1i = [xi ones(nidx,1)];
                A = zeros(ndo,ndi);
                b = zeros(ndo,1);
                for j = 1:ndo
                    coeffi = regress(yi(:,j), x1i);
                    A(j,:) = coeffi(1:end-1)';
                    b(j) = coeffi(end);
                end
                dyn = struct('A', A, 'b', b, 'idx', idx);
                cidx = (sbcl+obj.offset)./eps
                midx = obj.get_flat_idx(cidx);
                model{midx} = dyn;
            end
        end

        function dyn = get_old(obj, ci)
        dyn = obj.model{ci{:}};
        end

        function dyn = get(obj, k)
        %cell_idx = obj.get_cell_idx(c);
        %k = cell2mat(ci);
        midx = obj.get_flat_idx(k);
        dyn = obj.model{midx};
        end

        % TODO: UNFINISHED
        function get_mult_dim_idx()
        ci = cell(1,length(obj.la));
        [ci{:}] = ind2sub(midx, obj.la);
        end

        function fidx = get_flat_idx(obj, k)
        c = mat2cell(k,1,ones(1,length(k)));
        fidx = sub2ind(obj.la, c{:});
        end

        %function ndix = get_mult_dim_idx(obj, k, clar)
        %nidx = 0;
        %for i = 1:length(k)
        %    ndix(i) = ;
        %end
        %end

    end
end

function gridCells = flat_iter(cellRange, eps)
    nd = size(cellRange, 1);
    assert(size(cellRange, 2) == 2);
    H = cell(1,nd);
    rangeMat = cell(1,nd);
    for i = 1:nd
        rangeMat{i} = cellRange(i,1):eps(i):cellRange(i,2);
    end
    [H{:}] = ndgrid(rangeMat{:});
    gridCells = [];
    for i = 1:nd
        gridCells = [gridCells H{i}(:)];
    end
end

function s = mySign(X)
s = abs(X)./X;
nanIdx = isnan(s);
% replace sign(0) with 1
s(nanIdx) = 1; 
end

function X_snapped = snapToGrid(X,grid_eps)
grid_eps_mat = repmat(grid_eps,size(X,1),1);
X_snapped = mySign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat - grid_eps_mat/2);
X_snapped = round(X_snapped.*1e10)./1e10;
end

function gridCells = generateCellsFromRange(range, eps)
    cellRange = snapToGrid(range',eps)';
    dummyGridEps = eps;
    zeroGridEpsIdx = (eps == 0);
    dummyGridEps(zeroGridEpsIdx) = 1;
    % sanity check
    if range(zeroGridEpsIdx,1) ~= range(zeroGridEpsIdx,2)
        error('unhandled condition, grid_eps is 0 but range is non-zero measure')
    end

    numDim = size(cellRange,1);
    H = cell(1,numDim);
    rangeMat = cell(1,numDim);
    for i = 1:numDim
        rangeMat{i} = cellRange(i,1):dummyGridEps(i):cellRange(i,2);
    end
    [H{:}] = ndgrid(rangeMat{:});
    gridCells = [];
    for i = 1:numDim
        gridCells = [gridCells H{i}(:)];
    end

    gridCells = round(gridCells.*1e10)./1e10;
end

function model = Model_old(Range, eps, Y1, Y2, X)

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

function compare_models(old_model, new_model)
[r,c] = size(old_model);
assert(r*c == length(new_model.model));
for i = 1:r
    for j = 1:c
        smo = old_model{r,c};
        smn = new_model.get([r,c]);
        assert(isequal(smo.idx,smn.idx))
        assert(isequal(smo.A, smn.A))
        assert(isequal(smo.b, smn.b))
    end
end
fprintf('Old and New models are the same!\n');
end
