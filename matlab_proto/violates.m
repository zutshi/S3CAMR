function res = violates(s, prop)
res = ~isempty(Cube.getIntersection(s.x, prop));
end