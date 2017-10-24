function[f,g,cv,tfe] = CNC_machining(x,tfe)
if nargin == 0
	prob.nf = 2;
	prob.ng = 3;  
	prob.nx = 3;
    prob.range(1,:) = [250,400];
    prob.range(2,:) = [0.15,0.55];
    prob.range(3,:) = [0.5,6];
	f = prob;
else
	[f,g,cv,tfe] = metal_cutting2_true(x,tfe);
end
return


function [obj,g,cv,tfe] = metal_cutting2_true(x,tfe)
v = x(:,1);
f1 = x(:,2);
a = x(:,3);
r_n = 0.8; eta = 0.75; 
T = 5.48e9*(f1.^(-0.696).*v.^(-3.46).*a.^(-0.46));
MRR = 1000*f1.*v.*a;
F = 6.56e3*f1.^(0.917).*a.^(1.10).*v.^(-0.286);
P = v.*F./60000;
R = 125.*f1.^2/r_n;
obj(:,1) = 0.2 + 219912*(1 + (0.2./T))./MRR;
obj(:,2) = 219912*100./(MRR.*T);
g(:,1) =  P - eta*10;
g(:,2) = F - 5000;
g(:,3) = R - 50;
cv = sum(max(g,0),2);
tfe = tfe + size(x,1);
return