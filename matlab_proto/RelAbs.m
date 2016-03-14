classdef RelAbs
    
    properties
        model;
        %         GA;
        %offset;
        delta_t;
        eps;
    end
    
    %% abstraction functions
    methods
        %% Constructor
        function obj = RelAbs(m, delta_t, eps)
            obj.eps = eps;
            obj.model = m;
            %obj.offset = -(min(Range,[],2)' + Grid.grid_eps/2) + 1
            obj.delta_t = delta_t;
        end
        
        %%
        function sub_model = get_cell_model(obj, c)
            cell_idx = obj.model.cell2idx(c);
            sub_model = obj.model.get(cell_idx);
        end
        
        %% Simulator for the learned discrete dynamical system
        % Simulates from x for n steps through the system
        function [n,y] = simulate(obj, x, N, model_type, tol, dt)
            nd = size(x,2);
            if length(tol)>1
                assert(size(tol,1) == N);
                assert(size(tol,2) == nd);
                e = tol;
            else
                e = tol*(2*rand(N,nd)-1);
            end
            
            if strcmp(model_type, 'xt')
                %                 dt = 0.0099;
                
                n = (0:1:N) * dt;
                y = zeros(N+1,nd);
                % init y1
                y(1,:) = x;
                % compute yi
                for i = 1:1:N
                    yt = [y(i,:) dt];
                    
                    % map the point to itself in case the simulation trace
                    % reaches uncharted territory (unmodeled regions)
                    if Cube.check_sat(yt, obj.model.range) ~= 1
                        y(i+1,:) = y(i,:);
                    else
                        c = Grid.concrete2cell(yt, obj.eps);
                        sub_model = obj.get_cell_model(c);
                        %assert(all(sub_model.sbcl == c));
                        assert(sub_model.empty == false);
                        assert(all(yt' >= sub_model.P(:,1) & yt' <= sub_model.P(:,2)));
                        dyn = sub_model.M;
                        % y = A*x + b +- e
                        y(i+1,:) = (dyn.A * yt' + dyn.b + e(i,:)');
                    end
                end
                % strcmp(obj.model_type, 'x')
            else
                dt = obj.delta_t;
                n = (0:1:N) * dt;
                y = zeros(N+1,nd);
                
                % for debug
                %                 A = [];
                %                 B = [];
                
                % init y1
                y(1,:) = x;
                % compute yi
                for i = 1:1:N
                    c = Grid.concrete2cell(y(i,:), obj.eps);
                    sub_model = obj.get_cell_model(c);
                    dyn = sub_model.M;
                    %                     A = [A dyn.A];
                    %                     B = [B;dyn.b];
                    y(i+1,:) = dyn.A * y(i,:)' + dyn.b + e(i,:)';
                    %y(i+1,:) = fix(dyn.A*1000)/1000 * y(i,:)' + fix(dyn.b*1000)/1000;
                end
                % print the dynamics used
                %                 A'
                %                 B
            end
        end
        
        % multi time step - uniform randomly chosen
        function [n,y] = simulate_dmt(obj, x, tol, tss)
            nd = size(x,2);
            N = length(tss);
            if length(tol)>1
                assert(size(tol,1) == N);
                assert(size(tol,2) == nd);
                e = tol;
            else
                e = tol*(2*rand(N,nd)-1);
            end
                n = [0 cumsum(tss)];
                y = zeros(N+1,nd);
                y(1,:) = x;
                for i = 1:length(tss)
                    yt = [y(i,:) tss(i)];
                    if Cube.check_sat(yt, obj.model.range) ~= 1
                        y(i+1,:) = y(i,:);
                    else
                        c = Grid.concrete2cell(yt, obj.eps);
                        sub_model = obj.get_cell_model(c);
                        dyn = sub_model.M;
                        y(i+1,:) = (dyn.A * yt' + dyn.b + e(i,:)');
                    end
                end
            
            idx = randi(length(tss),1,1);
        end
        
        % Takes in the system model (PWA) and a state: rect hypercube
        % Returns a list of reachable rect hypercubes
        function S_ = compute_next_states(obj, s)
            S_ = {};
            % get all cells which intersect with the given hypercube
            [~,C] = Grid.generateCellsFromRange(s.x, obj.eps);
            
            % for each region of the cube which intersects a different cell
            % (assumes that every cell has different dynamics for convinience)
            for i = 1:size(C,1)
                c = C(i,:);
                crange = Grid.getCellRange(c, obj.eps);
                % Find the intersection with s
                intersection_cube = Cube.getIntersection(crange, s.x);
                assert(~isempty(intersection_cube));
                Cube.sanity_check_cube(intersection_cube);
                sub_model = obj.get_cell_model(c);
                dyn = sub_model.M;
                x_ = Cube.lin_transform_cube(intersection_cube, dyn);
                y_ = Cube.vertices2aligned_constraints(x_);
                S_ = [S_ struct('x', y_, 'n', s.n+1, 'path', [s.path; c])];
            end
        end
        
        % Takes in the system model (PWA) and a state: rect hypercube
        % Returns a list of reachable rect hypercubes
        function S_ = compute_next_cells(obj, s)
            S_ = {};
            % get all cells which intersect with the given hypercube
            %             [~,C] = Grid.generateCellsFromRange(s.x, obj.eps);
            
            % for each region of the cube which intersects a different cell
            % (assumes that every cell has different dynamics for convinience)
            
            % if the cube lies outside the range for which the pwa-model
            % was constructed, map it to the same cell
            if Cube.check_sat(s.c, obj.model.range) == 0
                s_ = struct('x', s.x, 'c', s.c, 'n', s.n+1, 'path', [s.path; s.c]);
                S_ = {s_};
                %return
            else
                sub_model = obj.get_cell_model(s.c);
                dyn = sub_model.M;
                x_ = Cube.lin_transform_cube(s.x, dyn);
                y_ = Cube.vertices2aligned_constraints(x_);
                
                [~,C_] = Grid.generateCellsFromRange(y_, obj.eps);
                
                for k = 1:size(C_,1)
                    c_ = C_(k,:);
                    crange_ = Grid.getCellRange(c_, obj.eps);
                    S_ = [S_ struct('x', crange_, 'c', c_, 'n', s.n+1, 'path', [s.path; s.c])];
                end
            end
        end
        
        %% Computes mean/max error between the given data and the
        % relationalized model
        function verify_model(obj, X, Y1, Y2)
            M = obj.model;
            [m,n] = size(M);
            for i = 1:m
                for j = 1:n
                    dyn = M{i,j};
                    k = dyn.idx;
                    diff = [Y1(k)'; Y2(k)'] - (dyn.A * X(k,:)' + repmat(dyn.b,1,length(k)));
                    error = sqrt(sum(diff.^2));
                    mean_error = mean(error);
                    max_error = max(error);
                    b1 = regress(Y1(k), [X(k,:), ones(length(k),1)])
                    b2 = regress(Y2(k), [X(k,:) ones(length(k),1)])
                    dyn.A
                    dyn.b
                    fprintf('mean error = %f, max_error = %f\n', mean_error, max_error);
                    fprintf('===========\n')
                end
            end
        end
    end
end
