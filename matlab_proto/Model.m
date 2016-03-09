classdef Model
    
    properties
        model;
        old_model;
        offset;
        % len_array
        la;
        model_type;
        eps;
        range;
    end
    methods
        function obj = Model(range, eps, Y, X, model_type)
            obj.range = range;
            obj.eps = eps;
            [rangeMat, sub_cells] = Grid.generateCellsFromRange(range, eps);
            r_ = [];
            for i = 1:length(rangeMat)
                r_ = [r_ length(rangeMat{i})];
            end
            
            obj.offset = -min(sub_cells);
            obj.la = r_;
            new_model = obj.get_model(sub_cells, eps, Y, X);
            obj.model = new_model;
            obj.model_type = model_type;
        end
        
        function model = get_model(obj, sub_cells, eps, Y, X)
            MIN_NUM_DATA_PTS = 10;
            
            % num_samples, input dim
            [ns, ndi] = size(X);
            % num_samples_Y, output dim
            [nsY, ndo] = size(Y);
            assert(ns == nsY);
            np = prod(obj.la);
            fprintf('number of paritions: %d\n', np);
            % create a flat array
            model = cell(1,np);
            
            textprogressbar('computing models: ');
            num_sub_cells = size(sub_cells,1);
            for i = 1:num_sub_cells
                % update progress
                textprogressbar(i*100/num_sub_cells);
                
                sbcl = sub_cells(i,:);
                crange = Grid.getCellRange(sbcl, eps);
                LB = crange(:, 1)';
                UB = crange(:, 2)';
                idx = find(all(X >= repmat(LB,ns,1) & X <= repmat(UB,ns,1), 2));
                midx = obj.cell2idx(sbcl);
                model{midx}.P = crange;
                model{midx}.sbcl = sbcl;
                if isempty(idx)
                    warning('empty idx')
                    model{midx}.empty = true;
                    continue
                end
                assert(length(idx)>MIN_NUM_DATA_PTS)
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
                model{midx}.M = dyn;
                model{midx}.empty = false;
                %                 model{midx}.idx = idx;
                
                %                 model{midx} = struct('P', crange, 'M', dyn, 'idx', idx);
            end
            textprogressbar('done');
        end
        
        function midx = cell2idx(obj, c)
            cidx = (c+obj.offset)./obj.eps  + 1;
            midx = obj.get_flat_idx(cidx);
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
            fidx = round(sub2ind(obj.la, c{:}));
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
            chk = [];
            M = obj.model;
            num_sub_models = length(M);
            flattened = cell(num_sub_models,1);
            for k = 1:num_sub_models
                m_k = M{k};
                if m_k.empty
                    continue
                end
                % translate the cube partition to a polyhedral partition
                pp = cube2poly(m_k.P);
                chk = [chk; m_k.P'];
                flattened{k}.P = pp;
                flattened{k}.M = m_k.M;
            end
            max(chk)
            min(chk)
        end
        
        
    end
end



% function model = Model_old(Range, eps, Y1, Y2, X)
% 
% [num_points, num_dims] = size(X);
% 
% nr = (Range(1,2) - Range(1,1))/eps(1);
% nc = (Range(2,2) - Range(2,1))/eps(2);
% num_paritions = nr*nc;
% model = cell(nr, nc);
% 
% % TODO: Replace by NDGRID
% i_ = 1;
% j_ = 1;
% for i = Range(1,1):eps(1):Range(1,2)-eps(1)
%     for j = Range(2,1):eps(2):Range(2,2)-eps(2)
%         idx = find(X(:,1) >= i & X(:,1) <= i+eps(1) & X(:,2) >= j & X(:,2) <= j+eps(2));
%         y1 = Y1(idx);
%         y2 = Y2(idx);
%         x = X(idx, :);
%         % append a column of ones to make the regress function compute
%         % b in Ax + b
%         x_ones = [x ones(size(x,1),1)];
%         b1 = regress(y1, x_ones);
%         b2 = regress(y2, x_ones);
%         %         A = [b1(1:end-1,1) b2(1:end-1,1)];
%         %         b = [b1(end); b2(end)];
%         A = [b1(1:end-1) b2(1:end-1)]';
%         b = [b1(end); b2(end)];
%         dyn = struct('A', A, 'b', b, 'idx', idx);
%         model{i_, j_} = dyn;
%         j_ = j_ + 1;
%     end
%     i_ = i_ + 1;
%     j_ = 1;
% end
% end
% 
% function compare_models(old_model, new_model)
% [r,c] = size(old_model);
% assert(r*c == length(new_model.model));
% for i = 1:r
%     for j = 1:c
%         smo = old_model{r,c};
%         smn = new_model.get([r,c]);
%         assert(isequal(smo.idx,smn.idx))
%         assert(isequal(smo.A, smn.A))
%         assert(isequal(smo.b, smn.b))
%     end
% end
% fprintf('Old and New models are the same!\n');
% end


%         function dyn = get_old(obj, ci)
%             dyn = obj.model{ci{:}};
%         end