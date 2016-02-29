function my_figure(h)
if disable_fig == 1
    return
end
% myFigure(h);
isFigureHandle = ishandle(h) && strcmp(get(h,'type'),'figure');
if isFigureHandle == 1 
    set(0,'CurrentFigure',h)
else
    figure(h);
end
end

function y = disable_fig()
y = 0;
end
