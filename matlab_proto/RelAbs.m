classdef RelAbs

    properties
        model;
        GA;
        %offset;
        delta_t;
    end

    %% abstraction functions
    methods
        %% Constructor
        function obj = RelAbs(m, ga, Range, delta_t)
            obj.model = m;
            obj.GA = ga;
            %obj.offset = -(min(Range,[],2)' + obj.GA.grid_eps/2) + 1
            obj.delta_t = delta_t;
        end

        %%
        function dyn = get_cell_dyn(obj, c)
            nd = length(c);
            cell_idx = obj.get_cell_idx(c);
            %ci = mat2cell(cell_idx, 1, ones(1, nd));
            dyn = obj.model.get(cell_idx);
        end

        %%
        function cell_idx = get_cell_idx(obj, c)
            cell_idx = fix((c + obj.model.offset)./obj.GA.grid_eps);
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
                y(i+1,:) = dyn.A * y(i,:)' + dyn.b;
            end
        end

        % Takes in the system model (PWA) and a state: rect hypercube
        % Returns a list of reachable rect hypercubes
        function S_ = compute_next_states(obj, s)
            S_ = {};
            % get all cells which intersect with the given hypercube
            C = obj.GA.generateCellsFromRange(s.x);

            % for each region of the cube which intersects a different cell
            % (assumes that every cell has different dynamics for convinience)
            for i = 1:size(C,1)
                c = C(i,:);
                crange = obj.GA.getCellRange(c);
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

        % dumos the model to a file
        % returns a flattened model
        % flattened_model = {{P0, M0},...,{Pi,Mi},...{Pn,Mn}}
        function model_flattened = get_flattened_model(obj)
            chk = [];
            m = obj.model.old_model;
            [mr, mc] = size(m);
            model_flattened = cell(mr*mc,1);
            k = 1;
            for i = 1:mr
                for j = 1:mc
                    dyn = m{i,j};
                    % ignore any othe rinfo in dyn but A and b
                    model = struct('A', dyn.A, 'b', dyn.b);
                    c = obj.GA.get_cell_from_idx([i,j], obj.model.offset);
                    crange = obj.GA.getCellRange(c);
                    chk = [chk; crange'];
                    p = cube2poly(crange);
                    model_flattened{k}.P = p;
                    model_flattened{k}.M = model;
                    k = k+1;
                end
            end
            max(chk)
            min(chk)
        end

    end
end
