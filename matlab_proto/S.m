%% Stack

classdef S < handle
    properties
        s;
    end
    %% abstraction functions
    methods
        % Constructor
        function obj = S()
            obj.s = {};
        end
        
        function push(obj, e)
            obj.s = [obj.s e];
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

