function [f,g,cv,tfe] = welded_beam(x,tfe)
if nargin == 0
	prob.nx = 4;
	prob.nf = 2;
	prob.ng = 4;
	prob.range(1,:) = [0.125,5.0];
	prob.range(2,:) = [0.1,10.0];
	prob.range(3,:) = [0.125,5.0];
	prob.range(4,:) = [0.1,10.0];
	f = prob;
else
	[f,g,cv,tfe] = welded_beam_true(x,tfe);
end
end

function [f,g,cv,tfe] = welded_beam_true(x,tfe)
h = x(:,1);
l = x(:,2);
t = x(:,3);
b = x(:,4);

delta = 2.1952./(b.*t.^3);
sigma = 504000./(b.*t.^2);
P_c = 64746.022.*(1 - 0.0282346.*t).*t.*b.^3;
tau_1 = 6000./(sqrt(2).*h.*l);
tau_2 = (6000.*(14 + 0.5.*l).*(sqrt(0.25.*(l.^2 + (h + t).^2))))./(2.*(0.707.*h.*l.*(l.^2./12 + 0.25.*(h + t).^2)));
tau = sqrt(tau_1.^2 + tau_2.^2 + (l.*tau_1.*tau_2)./(sqrt(0.25.*(l.^2 + (h + t).^2))));

g(:,1) = tau - 13600;
g(:,2) = sigma - 30000;
g(:,3) = h - b;
g(:,4) = 6000 - P_c;

f(:,1) = 1.10471.*h.^2.*l + 0.04811.*t.*b.*(14 + l);
f(:,2) = delta;
cv = sum(max(g,0),2);
tfe = tfe + size(x,1);
end
