%% CAUTION: Never run, never verified


%% Unique Q

% Q is implemented like a set: Guarenteed to contain unique entries
% Uniquemenss is defined using the ismember(e,Q,'rows') operator

% Difference from Q: The push() operation mimics a set add()

classdef UQ < handle
    properties
        q;
        history;
    end
    %% abstraction functions
    methods
        % Constructor
        function obj = UQ()
            obj.q = {};
            % TODO: num cols in history should be num dim
            obj.history = [nan nan];
        end
        
        function push(obj, e)
            % check unique: has it ever been pushed before?
            if ~ismember(e.c,obj.history,'rows')
                obj.q = [obj.q e];
                obj.history = [obj.history; e.c];
            end
        end        
        
        % No checks!
        function e = pop(obj)
            e = obj.q{1};
            obj.q(1) = [];
        end
        
        function res = empty(obj)
            res = isempty(obj.q);
        end
        
    end
end

