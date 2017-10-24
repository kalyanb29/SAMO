function [f,g,cv,tfe] = tool_spindle_design(x,tfe)
	if nargin == 0
		prob.nx = 4;
		prob.nf = 2;
		prob.ng = 3;
		prob.range(1,:) = [150,200];
		prob.range(2,:) = [25,72];
		prob.range(3,:) = [1,4];
		prob.range(4,:) = [1,4];
		f = prob;
	else
		[f,g,cv,tfe] = tool_spindle_true(x,tfe);
	end
end


function [f,g,cv,tfe] = tool_spindle_true(x,tfe)
x_3 = round(x(:,3));
x_4 = round(x(:,4));
d_a_set = [80 85 90 95];
d_b_set = [75 80 85 90];
d_a = zeros(size(x,1),1);
d_b = zeros(size(x,1),1);
for i = 1:size(x,1)
    d_a(i) = d_a_set(x_3(i));
    d_b(i) = d_b_set(x_4(i));
end
% Var = l,d_o,d_a,d_b
    d_om = 25;
    d_a1 = 80;
    d_a2 = 95;
    d_b1 = 75;
    d_b2 = 90;
    p_1 = 1.25;
    p_2 = 1.05;
    l_k = 150;
    l_g = 200;
    a = 80;
    E = 210000;
    F = 10000;
    del_a = 0.0054;
    del_b = -0.0054;
    del = 0.01;
    del_ra = -0.001;
    del_rb = -0.001;
    I_a = 0.049*(d_a.^4 - x(:,2).^4);
    I_b = 0.049*(d_b.^4 - x(:,2).^4);
    c_a = 35400*abs(del_ra)^(1/9)*d_a.^(10/9);
    c_b = 35400*abs(del_rb)^(1/9)*d_b.^(10/9);
	f(:,1) = pi/4*(a*(d_a.^2 - x(:,2).^2) + x(:,1).*(d_b.^2 - x(:,2).^2));
	f(:,2) = F*a^3./(3*E.*I_a).*(1 + (x(:,1).*I_a)./(a*I_b)) + (F./c_a).*((1 + a./x(:,1)).^2 + (c_a*a^2)./(c_b.*x(:,1).^2));
    
    g(:,1) = p_1*x(:,2) - d_b;
	g(:,2) = p_2*d_b - d_a;
	g(:,3) = abs(del_a + (del_a - del_b)*a./x(:,1)) - del;
    cv = sum(max(g,0),2);
    tfe = tfe + size(x,1);
end
