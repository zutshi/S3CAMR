% Provides functions to construct standardized file names used accross the
% project

classdef FileNames
    
    methods(Static)
        function y = model_file_name(SYS_NAME)
            y = ['../examples/vdp/data/matlab_model/' SYS_NAME '_flat_model'];
        end
        
        function y = all_file_name(SYS_NAME)
            y = ['../examples/vdp/data/matlab_model/' SYS_NAME '_all'];
        end
        
        function y = ip_file_name(SYS_NAME)
            y = ['../examples/vdp/data/sim_data/' SYS_NAME '_data.mat'];
        end
        
    end
    
end