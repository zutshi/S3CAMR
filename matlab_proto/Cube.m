classdef Cube
    methods (Static)
        
        function sanity_check_cube(c)
            assert(all(c(:, 2) - c(:, 1)>=0));
        end
        function icube = getIntersection(cube1,cube2)
            
            I = [max(cube1(:,1),cube2(:,1)) min(cube1(:,2),cube2(:,2))];
            diff = I(:,2) - I(:,1);
            % diff = fix(diff*1e8)/1e8;
            if all(diff >= 0)
                icube = I;
            else
                % empty intersection
                icube = [];
            end
        end
        
        % returns vertices of a cell
        function v = get_vertices(c)
            % nd = size(c,1);
            % H = cell(1,nd);
            % [H{:}] = ndgrid(c(1,:), c(2,:));
            % V = [H{:}(:)];  % Can not figure out this step!! {:}(:) is not legal
            
            % Much easier! Requires NN toolbox?
            v = combvec(c(1,:), c(2,:))';
        end
        
        % Over-approximates the linear transform of a hyper rectangle by
        % another hyper rectangle
        
        function C_ = lin_transform_cube(C, dyn)
            vertices = Cube.get_vertices(C); % [c1; c2; c3; ...; cn]
            [num_corners, nd] = size(vertices);
            x = reshape(vertices', nd, num_corners);
            x_ = dyn.A*x + repmat(dyn.b, 1, size(x,2));
            C_ = reshape(x_', num_corners, nd);
        end
        
        % Overaproximates an oriented cube by an aligned cube
        % Takes in the vertices
        function acube = vertices2aligned_constraints(ocubev)
            acube = [min(ocubev)' max(ocubev)'];
        end
        
        % does the coordinate x lie in cube?
        function sat = check_sat(x,cube)
            sat = all(cube(:,1) <= x' & cube(:,2) >= x');
        end
        
    end
end