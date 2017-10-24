function [f,g,cv,tfe] = zdt1(x,tfe)
	if nargin == 0
		prob.nx = 10;
		prob.nf = 2;
		prob.ng = 0;
		for i = 1:prob.nx
			prob.range(i,:) = [0.0, 1.0];
		end
		f = prob;
	else
		[f,g,cv,tfe] = zdt1_true(x,tfe);
	end
end


function [f,c,cv,tfe] = zdt1_true(x,tfe)
	N = size(x,2);
	f(:,1) = x(:,1);
	g = 1 + 9/(N-1) * sum(x(:,2:N),2);
	h = 1 - sqrt(f(:,1)./g);
	f(:,2) = g.*h;
	c = [];
    cv = zeros(size(x,1),1);
    tfe = tfe + size(x,1);
end
