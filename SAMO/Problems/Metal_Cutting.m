function[f,g,cv,tfe] = Metal_Cutting(x,tfe)
if nargin == 0
	prob.nf = 3;
	prob.ng = 2;  
	prob.nx = 3;
    prob.range(1,:) = [70,90];
    prob.range(2,:) = [0.1,2];
    prob.range(3,:) = [0.1,5];
	f = prob;
else
	[f,g,cv,tfe] = metal_cutting3_true(x,tfe);
end
return


function [obj,g,cv,tfe] = metal_cutting3_true(x,tfe)
v = x(:,1);
f1 = x(:,2);
a = x(:,3);
T = 1575134.21*(f1.^(-1.55).*v.^(-1.70).*a.^(-1.22));
MRR = 1000*f1.*v.*a;
F = 1.38*f1.^(1.18).*a.^(1.26);
P = 0.000626*v.*f1.^(0.24).*a.^(0.11);
obj(:,1) = 0.12 + 2313.76*(1 + (0.26./T))./MRR;
obj(:,2) = ((13.55./T) + 0.39).*obj(:,1);
obj(:,3) = 0.0088.*v + 0.3232.*f1 + 0.3144.*a;
g(:,1) =  P - 5;
g(:,2) = F - 230;
cv = sum(max(g,0),2);
tfe = tfe + size(x,1);
return