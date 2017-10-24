function[f,g,cv,tfe] = Car_Crash(x,tfe)
if nargin == 0
	prob.nf = 3;
	prob.ng = 0;  
	prob.nx = 5;
    for i = 1:prob.nx
        prob.range(i,:) = [1,3];
    end
	f = prob;
else
	[f,g,cv,tfe] = Car_Crash_true(x,tfe);
end
return


function [f,g,cv,tfe] = Car_Crash_true(x,tfe)
f(:,1) = 1640.2823 + 2.3573285.*x(:,1) + 2.3220035.*x(:,2) + 4.5688768*x(:,3) + 7.7213633.*x(:,4) + 4.4559504.*x(:,5);
f(:,2) = 6.5856 + 1.15.*x(:,1) - 1.0427.*x(:,2) + 0.9738.*x(:,3) + 0.8364.*x(:,4) - 0.3695.*x(:,1).*x(:,4) + 0.0861.*x(:,1).*x(:,5) + 0.3628.*x(:,2).*x(:,4) - 0.1106.*x(:,1).^2 - 0.3437.*x(:,3).^2 + 0.1764.*x(:,4).^2;
f(:,3) = -0.0551 + 0.0181.*x(:,1) + 0.1024.*x(:,2) + 0.0421.*x(:,3) - 0.0073.*x(:,1).*x(:,2) + 0.024.*x(:,2).*x(:,3) - 0.0118.*x(:,2).*x(:,4) - 0.0204.*x(:,3).*x(:,4) - 0.008.*x(:,3).*x(:,5) - 0.0241.*x(:,2).^2 + 0.0109.*x(:,4).^2;
g = [];
cv = zeros(size(x,1),1);
tfe = tfe + size(x,1);
return