function plot_cell(cube, color)
global PLT;
global VERBOSE;
% if PLT
    line([cube(1,1) cube(1,2) cube(1,2) cube(1,1) cube(1,1)],[cube(2,1) cube(2,1) cube(2,2) cube(2,2) cube(2,1)],'color',color);
    if VERBOSE
        drawnow
    end
% end
end