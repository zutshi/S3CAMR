function [t,y] = vdp_sim(x0, Tspan)

optionsODE = odeset('Refine',4,'RelTol',1e-6);
[t,y] = ode45(@vdp_dyn,Tspan,x0,optionsODE);
end


function Y = vdp_dyn(~,X)
Y(1) = X(2);
Y(2) = 5 * (1 - X(1)^2) * X(2) - X(1);
Y = Y';
end