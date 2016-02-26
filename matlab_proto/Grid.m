classdef Grid
    
    methods(Static)
        
        
        function X_snapped = snapToGrid(X,grid_eps)
            grid_eps_mat = repmat(grid_eps,size(X,1),1);
            
            X_snapped = mySign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat + grid_eps_mat/2);
            
            %% restore the states that were not supposed to be snapped
            nanIdx = isnan(X_snapped);
            X_snapped(nanIdx) = X(nanIdx);
            
            X_snapped = round(X_snapped.*1e10)./1e10;
        end
        
        % Concrete state: p(x,y) -> Abstract state a[x1 x2; y1 y2]
        function a = concrete2abs(p, grid_eps)
            a = snapToGrid(p, grid_eps);
        end
        
        %%
        % returns cells as
        % [c0]
        % [c1]
        % [..]
        % [cn]
        function [rangeMat, gridCells] = generateCellsFromRange(range, eps)
            % cellRange = expand_interval(snapToGrid(range',eps), eps);
            warning('underapproximating!')
            cellRange = Grid.snapToGrid(range',eps)';
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
        function cellRange = getCellRange(X, grid_eps)
            % Remove the warnign after checking all getCellRange() calls in
            % the code.
            %             warning('Just returns a +- eps. Please make sure that is the intended usage!.')
            cellRange = [(X - grid_eps/2)' (X + grid_eps/2)'];
        end
        
        function c = get_cell_from_idx(idx, offset, grid_eps)
            c = idx .* grid_eps; % - obj.grid_eps/2;
            c = c - offset;
        end
        
        function c = concrete2cell(x, grid_eps)
            c = Grid.snapToGrid(x,grid_eps);
        end
        
        
        % function X_snapped = snapToGrid(X,grid_eps)
        % grid_eps_mat = repmat(grid_eps,size(X,1),1);
        % X_snapped = mySign(X).*(fix(abs(X)./grid_eps_mat).*grid_eps_mat + grid_eps_mat/2);
        % X_snapped = round(X_snapped.*1e10)./1e10;
        % end
        
        %         function gridCells = generateCellsFromRange(range, grid_eps)
        %             cellRange = snapToGrid(range',grid_eps)';
        %             dummyGridEps = grid_eps;
        %             zeroGridEpsIdx = (grid_eps == 0);
        %             dummyGridEps(zeroGridEpsIdx) = 1;
        %             % sanity check
        %             if range(zeroGridEpsIdx,1) ~= range(zeroGridEpsIdx,2)
        %                 error('unhandled condition, grid_eps is 0 but range is non-zero measure')
        %             end
        %
        %             numDim = size(cellRange,1);
        %             H = cell(1,numDim);
        %             rangeMat = cell(1,numDim);
        %             for i = 1:numDim
        %                 rangeMat{i} = cellRange(i,1):dummyGridEps(i):cellRange(i,2);
        %             end
        %             [H{:}] = ndgrid(rangeMat{:});
        %             gridCells = [];
        %             for i = 1:numDim
        %                 gridCells = [gridCells H{i}(:)];
        %             end
        %
        %             gridCells = round(gridCells.*1e10)./1e10;
        %         end
        
        
    end
end