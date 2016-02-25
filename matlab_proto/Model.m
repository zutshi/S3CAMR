classdef Model
    
    properties
        model;
        old_model;
        offset;
        % len_array
        la;
    end
    methods
        function obj = Model(range, eps, Y, X, model_type)
            sub_cells = generateCellsFromRange(range, eps);
            %             max(sub_cells)
            %             min(sub_cells)
            
            % num of elements in each dim
            cmin = snapToGrid(range(:,1)', eps);
            cmax = snapToGrid(range(:,2)', eps);
            % num_paritions
            r_ = (cmax-cmin)./eps + 1;
            obj.offset = -min(sub_cells) + 1;
            obj.la = r_;
            new_model = obj.Model_new(sub_cells, eps, Y, X);
            obj.model = new_model;
            obj.model_type = model_type;
        end
        
        function model = Model_new(obj, sub_cells, eps, Y, X)
            % num_samples, input dim
            [ns, ndi] = size(X);
            % num_samples_Y, output dim
            [nsY, ndo] = size(Y);
            assert(ns == nsY);
            np = prod(obj.la);
            fprintf('number of paritions: %d\n', np);
            % create a flat array
            model = cell(1,np);
            
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
                dyn = struct('A', A, 'b', b);
                cidx = (sbcl+obj.offset)./eps;
                midx = obj.get_flat_idx(cidx);
                crange = getCellRange(sbcl, eps);
                model{midx} = struct('P', crange, 'M', dyn, 'idx', idx);
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
        function get_mult_dim_idx(obj)
            ci = cell(1,length(obj.la));
            [ci{:}] = ind2sub(midx, obj.la);
        end
        
        function fidx = get_flat_idx(obj, k)
            %         c = mat2cell(k,1,ones(1,length(k)));
            c = num2cell(k);            
            fidx = sub2ind(obj.la, c{:});
        end
        
        %function ndix = get_mult_dim_idx(obj, k, clar)
        %nidx = 0;
        %for i = 1:length(k)
        %    ndix(i) = ;
        %end
        %end
        
        % get a flattened model to dump a file
        % flattened_model = {{P0, M0},...,{Pi,Mi},...{Pn,Mn}}
        function flattened = flat(obj)
%             chk = [];
            M = obj.model;
            num_sub_models = length(M);
            flattened = cell(1,num_sub_models);
            for k = 1:num_sub_models
                m_k = M{k};
                % translate the cube partition to a polyhedral partition
                pp = cube2poly(m_k.P);
%                 chk = [chk; m_k.P'];
                flattened{k}.P = pp;
                flattened{k}.M = m_k.M;
            end
%             max(chk);
%             min(chk);
        end
        %% Simulator for the learned discrete dynamical system
        % Simulates from x for n steps through the system
        function [n,y] = simulate(obj, x, N)
            dt = obj.delta_t;
            nd = length(x);
            n = 0:1:N * dt;
            y = zeros(N+1,nd);
            
            % init y1
            y(1,:) = x;
            % compute yi
            for i = 1:1:N
                c = obj.GA.concrete2cell(y(i,:));
                dyn = obj.get_cell_dyn(c);
                y(i+1,:) = fix(dyn.A*1000)/1000 * y(i,:)' + fix(dyn.b*1000)/1000;
            end
        end
    end
end

function cellRange = getCellRange(X, eps)
cellRange = [(X - eps/2)' (X + eps/2)'];
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
X_snapped = mySign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat + grid_eps_mat/2);
X_snapped = round(X_snapped.*1e10)./1e10;
end

function gridCells = generateCellsFromRange(range, eps)
% cellRange = expand_interval(snapToGrid(range',eps), eps);
warning('underapproximating!')
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
