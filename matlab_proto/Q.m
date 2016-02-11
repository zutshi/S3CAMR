classdef Q < handle
    properties
        q;
    end
    %% abstraction functions
    methods
        % Constructor
        function obj = Q()
            obj.q = {};
        end
        
        function push(obj, e)
            obj.q = [obj.q e];
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

