function[f,g,cv,tfe] = bulk_carrier_design(x,tfe)
if nargin == 0
		prob.nx = 6;
		prob.nf = 3;
		prob.ng = 9;
		prob.range(1,:) = [60, 600];
        prob.range(2,:) = [10, 100];
        prob.range(3,:) = [4, 40];
        prob.range(4,:) = [3, 30];
        prob.range(5,:) = [0.63, 0.75];
        prob.range(6,:) = [14, 18];
		f = prob;
	else
		[f,g,cv,tfe] = bulk_carrier_design_true(x,tfe);
	end
end


function [f,c,cv,tfe] = bulk_carrier_design_true(x,tfe)
	L = x(:,1);
    B = x(:,2);
    D = x(:,3);
    T = x(:,4);
    C_B = x(:,5);
    V_k = x(:,6);
    W_s = 0.034*L.^1.7.*B.^0.7.*D.^0.4.*C_B.^0.5;
    W_o = L.^0.8.*B.^0.6.*D.^0.3.*C_B.^0.1;
    a = 4977.06*C_B.^2 - 8105.61*C_B + 4456.51;
    b = -10847.2*C_B.^2 + 12817*C_B - 6960.32;
    Fn = (0.5144*V_k)./(sqrt(9.8065*L));
    Del = 1.025*L.*B.*T.*C_B;
    P = (Del.^(2/3).*V_k.^3)./(a+b.*Fn);
    W_M = 0.17*P.^0.9;
    LS = W_s + W_o + W_M;
    DW = Del - LS;
    DC = 0.2 + (0.19*24)*P/1000;
    D_s = 5000./(24*V_k);
    FC = DC.*(D_s + 5);
    CSW = 2*DW.^0.5;
    DW_c = DW - FC - CSW;
    D_p = 2*(DW_c/8000) + 1;
    RTPA = 350./(D_s + D_p);
    C_s = 1.3*(2000*W_s.^0.85 + 3500*W_o + 2400*P.^0.8);
    C = 0.2*C_s;
    C_R = 40000*DW.^0.3;
    C_v = 1.05*DC.*D_s*100 + 6.3*DW.^0.8;
    C_A = C + C_R + C_v.*RTPA;
    AC = DW_c.*RTPA;
    f(:,1) = LS;
    f(:,2) = -DW_c.*RTPA;
    f(:,3) = C_A./AC;
    c(:,1) = 6 - L./B;
    c(:,2) = L./D - 15;
    c(:,3) = L./T - 19;
    c(:,4) = Fn - 0.32;
    c(:,5) = DW - 500000;
    c(:,6) = 3000 - DW;
    c(:,7) = T - 0.45*DW.^0.31;
    c(:,8) = T - 0.7*D - 0.7;
    c(:,9) = 0.07*B - 0.53*T - ((0.085*C_B - 0.002).*B.^2)./(T.*C_B) + 1 + 0.52*D;
    id1 = find(imag(f(:,1)));
    id2 = find(imag(f(:,2)));
    id3 = find(imag(f(:,3)));
    id11 = union(id1,id2);
    id = union(id11,id3);
    f(id,:) = repmat(1e10*[1 -1 1],numel(id),1);
    c(id,:) = repmat(1e10*ones(1,9),numel(id),1);
    cv = sum(max(c,0),2);
    tfe = tfe + size(x,1);
end