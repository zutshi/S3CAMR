classdef GridAbstraction

    properties
        grid_eps;
    end

    %% abstraction functions
    methods
        %% Constructor
        function obj = GridAbstraction(e)
            obj.grid_eps = e;
        end

        % Concrete state: p(x,y) -> Abstract state a[x1 x2; y1 y2]
        function a = concrete2abs(obj, p)
            a = snapToGrid(p, obj.grid_eps);
        end

        %%
        % returns cells as
        % [c0]
        % [c1]
        % [..]
        % [cn]
        function gridCells = generateCellsFromRange(obj, range)
            eps = obj.grid_eps;
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

        %%
        function cellRange = getCellRange(obj, X)
            % Remove the warnign after checking all getCellRange() calls in
            % the code.
            warning('Just returns a +- eps. Please make sure that is the intended usage!.')
            cellRange = [(X - obj.grid_eps/2)' (X + obj.grid_eps/2)'];
        end

        function c = get_cell_from_idx(obj, idx, offset)
            c = idx .* obj.grid_eps; % - obj.grid_eps/2;
            c = c - offset;
        end

        function c = concrete2cell(obj, x)
            c = snapToGrid(x,obj.grid_eps);
        end
    end
end



function X_snapped = snapToGrid(X,grid_eps)
grid_eps_mat = repmat(grid_eps,size(X,1),1);

X_snapped = mySign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat + grid_eps_mat/2);

%% restore the states that were not supposed to be snapped
nanIdx = isnan(X_snapped);
X_snapped(nanIdx) = X(nanIdx);

X_snapped = round(X_snapped.*1e10)./1e10;
end


% Matlab's sign() doesn't handle sign(0) properly! returns 0!!
% X_snapped = sign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat + grid_eps_mat/2);
% this one returns sign(0) = 1
function s = mySign(X)
s = abs(X)./X;
nanIdx = isnan(s);
% replace sign(0) with 1
s(nanIdx) = 1;
end
