function plot_cell(cube, color, opts)
% global PLT;
VERBOSE = opts.v;
% if PLT
    line([cube(1,1) cube(1,2) cube(1,2) cube(1,1) cube(1,1)],[cube(2,1) cube(2,1) cube(2,2) cube(2,2) cube(2,1)],'color',color);
    if VERBOSE > 2
        drawnow
    end
% end
end
