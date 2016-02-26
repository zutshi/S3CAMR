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
        function [n,y] = simulate(obj, x, TN, model_type)
            if strcmp(model_type, 'xt')
                N = TN;
                
                dt = 0.01; %obj.delta_t;
                
                
                nd = size(x,2);
                n = (0:1:N) * dt;
                y = zeros(N+1,nd);
                % init y1
                y(1,:) = x;
                % compute yi
                for i = 1:1:N
                    %y(i,:)
                    yt = [y(i,:) dt];
                    c = Grid.concrete2cell(yt, obj.eps);
                    sub_model = obj.get_cell_model(c);
                    %                     assert(all(sub_model.sbcl == c));
                    assert(sub_model.empty == false);
                    assert(all(yt' >= sub_model.P(:,1) & yt' <= sub_model.P(:,2)));
                    dyn = sub_model.M;
                    y(i+1,:) = (dyn.A * yt' + dyn.b);
                end
            else% strcmp(obj.model_type, 'x')
                N = TN;
                dt = obj.delta_t;
                nd = length(x);
                n = (0:1:N) * dt;
                y = zeros(N+1,nd);
                
                % init y1
                y(1,:) = x;
                % compute yi
                for i = 1:1:N
                    c = Grid.concrete2cell(y(i,:), obj.eps);
                    sub_model = obj.get_cell_model(c);
                    dyn = sub_model.M;
                    y(i+1,:) = fix(dyn.A*1000)/1000 * y(i,:)' + fix(dyn.b*1000)/1000;
                end
            end
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
                dyn = obj.get_cell_dyn(c);
                x_ = Cube.lin_transform_cube(intersection_cube, dyn);
                y_ = Cube.vertices2aligned_constraints(x_);
                S_ = [S_ struct('x', y_, 'n', s.n+1, 'path', [s.path; c])];
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
