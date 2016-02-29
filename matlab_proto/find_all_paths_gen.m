
%% Generator version of find_all_paths_thresh()
% TODO
% Pointless to mantain two versions if the generator version is not much
% slower. Verify and take care.

function cn = find_all_paths_gen(RA, N, x0, prop, opts)
VERBOSE = opts.v;
PLT = opts.p;

% states reached in n steps
final_states = {};

% x0 = [-1  0  0.4  % -x1 +   + 0.4 <= 0
%        0 -1  0.4  %     -x2 + 0.4 <= 0
%        1  0 -0.4  % x1 +    - 0.4 <= 0
%        0  1 -0.4  %    + x2 - 0.4 <= 0
%       ];

S0.x = x0;
S0.n = 0;
% TODO
% something is wrong with using the below. The paths still turn out to be
% a double array and not a cell array!
% S0.path = {};
% Suspect that the something funny is going on in compute_next_states(), at
% S_ = [S_ struct('x', y_, 'n', s.n+1, 'path', [s.path; c])];
S0.path = [];

% workq = Q();
workq = S();

% Push all initial states
workq.push(S0);

if PLT
    my_figure(1)
    axis([-2,2, -8, 8])
    hold on
    colors = ['r','b','k','g','y','m'];
    cidx = 0;
end

    function violating_path = iter()
        while ~workq.empty()
            s = workq.pop();
            
            if VERBOSE > 3
                fprintf('parent state: pausing...\n');
                pause();
            end
            if PLT
                cidx = 1+mod(cidx, length(colors));
                plot_cell(s.x,colors(cidx),opts);
            end
            
    S_ = RA.compute_next_states(s);
            for i = 1:length(S_);
                s_ = S_{i};
                
                if VERBOSE > 3
                    fprintf('child state: pausing...\n');
                    pause();
                end
                if PLT
                    plot_cell(s_.x,colors(cidx),opts);
                end
                
                
                Cube.sanity_check_cube(s_.x);
                if s_.n < N
                    workq.push(s_);
                else
                    final_states = [final_states s_];
                end
                if violates(s_, prop)
                    display('violation found')
                    % strip away the first cell, it will be later replaced by x0
                    s_.path(1,:) = [];
                    violating_path = s_.path;
                    return;
                end
            end
        end
    end
cn = @iter;
end
