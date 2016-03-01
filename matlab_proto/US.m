%% Unique Stack

% S is implemented like a set: Guarenteed to contain unique entries
% Uniquemenss is defined using the ismember(e,S,'rows') operator

% Difference from S: The push() operation mimics a set add()


classdef US < handle
    properties
        s;
        history;
    end
    %% abstraction functions
    methods
        % Constructor
        function obj = US()
            obj.s = {};
            
            % TODO: num cols in history should be num dim
            obj.history = [nan nan];
        end
        
        function push(obj, e)
            % check unique: has it ever been pushed before?
            if ~ismember(e.c,obj.history,'rows')
                obj.s = [obj.s e];
                obj.history = [obj.history; e.c];
            end
        end
        
        % No checks!
        function e = pop(obj)
            e = obj.s{end};
            obj.s(end) = [];
        end
        
        function res = empty(obj)
            res = isempty(obj.s);
        end
        
    end
end

