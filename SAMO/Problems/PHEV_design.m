function[f,g,cv,tfe] = PHEV_design(x,tfe)
% Inputs:
% Objnum = number of objectives (3)
% modelnum = 1: Peterson battery degradation model, 2: Rosenkranz model
% batterytype = 1: Li-ion battery, 2: NiMH battery
% batteryreplace = 1: Battery leasing 2: Buy lease
% flag = 1: All, 2: Individual single distance, 3: All single distance ,4: all upto that distance
if nargin == 0
    prob.nx = 6;
    prob.nf = 3;
    prob.ng = 3;
    prob.range(1,:) = [30/57,60/57];  % Engine scaling factor
    prob.range(2,:) = [50/52,110/52]; % Motor scaling factor
    prob.range(3,:) = [0.2,1];        % Battery pack scaling factor
    prob.range(4,:) = [0,0.8];        % Battery energy swing
    prob.range(5,:) = [50,50];        % Cutoff point
    prob.range(6,:) = [1,1];          % Combination of type of cars: 1 = PHEV+HEV, 2 = PHEV+CV, 3 = HEV+CV, 4 = HEV+PHEV, 5 = CV+PHEV, 6=CV+HEV in respectively opposite sequence
    f = prob;
else
    [f,g,cv,tfe] = PHEV_design_true(x,tfe);
end
end

function[f,g,cv,tfe] = PHEV_design_true(x,tfe)
flagdist = 50;
maxdist = 200;
% maxdist = 25; % maxdist is maximum range of prob.range(5,:)
flag = 2; % Individual setting
% flagdist = 3; % All/Single drivers drives this distance per day
lambda = 0.0296;
N = size(x,1);
sequence = round(x(:,6));
modelnum = 1;
batterytype = 1;
batteryreplace = 2;
for i = 1:numel(sequence)
    if sequence(i) < 1
        sequence(i) = 1;
    elseif sequence(i) > 6
        sequence(i) = 6;
    end
end
cutoffpoint = x(:,5);
for i = 1:N
    sequencei = sequence(i);
    cutoffdist = cutoffpoint(i);
    if sequencei ==  1 || sequencei ==  4
        [f_PHEV_G,f_PHEV_V,f_PHEV_C,g_PHEV,distance_PHEV] = calculate_objective_PHEV(x(i,:),modelnum,batterytype,batteryreplace,cutoffdist,sequencei,maxdist,flag,flagdist,lambda);
        [f_HEV_G,f_HEV_V,f_HEV_C,g_HEV,distance_HEV] = calculate_objective_HEV(x(i,:),cutoffdist,sequencei,maxdist,flag,flagdist,lambda);
        if sequencei == 1
            if flag == 1
                f_G1 = [f_PHEV_G f_HEV_G];
                f_V1 = [f_PHEV_V f_HEV_V];
                f_C1 = [f_PHEV_C f_HEV_C];
                distance = [distance_PHEV distance_HEV];
                g(i,:) = g_PHEV + g_HEV;
            else
                f_G1 = [f_PHEV_G];
                f_V1 = [f_PHEV_V];
                f_C1 = [f_PHEV_C];
                distance = [distance_PHEV];
                g(i,:) = g_PHEV;
            end
        else
            if flag == 1
                f_G1 = [f_HEV_G f_PHEV_G];
                f_V1 = [f_HEV_V f_PHEV_V];
                f_C1 = [f_HEV_C f_PHEV_C];
                distance = [distance_HEV distance_PHEV];
                g(i,:) = g_HEV + g_PHEV;
            else
                f_G1 = [f_HEV_G];
                f_V1 = [f_HEV_V];
                f_C1 = [f_HEV_C];
                distance = [distance_HEV];
                g(i,:) = g_HEV;
            end
        end
    elseif sequencei ==  2 || sequencei ==  5
        [f_PHEV_G,f_PHEV_V,f_PHEV_C,g_PHEV,distance_PHEV] = calculate_objective_PHEV(x(i,:),modelnum,batterytype,batteryreplace,cutoffdist,sequencei,maxdist,flag,flagdist,lambda);
        [f_CV_G,f_CV_V,f_CV_C,g_CV,distance_CV] = calculate_objective_CV(x(i,:),cutoffdist,sequencei,maxdist,flag,flagdist,lambda);
        if sequencei == 2
            if flag == 1
                distance = [distance_PHEV distance_CV];
                f_G1 = [f_PHEV_G f_CV_G];
                f_V1 = [f_PHEV_V f_CV_V];
                f_C1 = [f_PHEV_C f_CV_C];
                g(i,:) = g_PHEV + g_CV;
            else
                distance = [distance_PHEV];
                f_G1 = [f_PHEV_G];
                f_V1 = [f_PHEV_V];
                f_C1 = [f_PHEV_C];
                g(i,:) = g_PHEV;
            end
        else
            if flag == 1
                distance = [distance_CV distance_PHEV];
                f_G1 = [f_CV_G f_PHEV_G];
                f_V1 = [f_CV_V f_PHEV_V];
                f_C1 = [f_CV_C f_PHEV_C];
                g(i,:) = g_CV + g_PHEV;
            else
                distance = [distance_PHEV];
                f_G1 = [f_CV_G];
                f_V1 = [f_CV_V];
                f_C1 = [f_CV_C];
                g(i,:) = g_CV;
            end
        end
    else
        [f_HEV_G,f_HEV_V,f_HEV_C,g_HEV,distance_HEV] = calculate_objective_HEV(x(i,:),cutoffdist,sequencei,maxdist,flag,flagdist,lambda);
        [f_CV_G,f_CV_V,f_CV_C,g_CV,distance_CV] = calculate_objective_CV(x(i,:),cutoffdist,sequencei,maxdist,flag,flagdist,lambda);
        if sequencei == 3
            if flag == 1
                distance = [distance_HEV distance_CV];
                f_G1 = [f_HEV_G f_CV_G];
                f_V1 = [f_HEV_V f_CV_V];
                f_C1 = [f_HEV_C f_CV_C];
                g(i,:) = g_HEV + g_CV;
            else
                distance = [distance_HEV];
                f_G1 = [f_HEV_G];
                f_V1 = [f_HEV_V];
                f_C1 = [f_HEV_C];
                g(i,:) = g_CV;
            end
        else
            if flag == 1
                distance = [distance_CV distance_HEV];
                f_G1 = [f_CV_G f_HEV_G];
                f_V1 = [f_CV_V f_HEV_V];
                f_C1 = [f_CV_C f_HEV_C];
                g(i,:) = g_CV + g_HEV;
            else
                distance = [distance_CV];
                f_G1 = [f_CV_G];
                f_V1 = [f_CV_V];
                f_C1 = [f_CV_C];
                g(i,:) = g_CV;
            end
        end
    end
    f_G(i,:) = f_G1;
    f_V(i,:) = f_V1;
    f_C(i,:) = f_C1;
    distanceall(i,:) = distance;
end
f = zeros(size(x,1),3);
rownotnan = [];
for i = 1:size(x,1)
    if flag == 1 || flag == 4
        f(i,1) = (trapz(distanceall(i,~isnan(f_G(i,:))),f_G(i,~isnan(f_G(i,:))),2));
        f(i,2) = (trapz(distanceall(i,~isnan(f_V(i,:))),f_V(i,~isnan(f_V(i,:))),2));
        f(i,3) = (trapz(distanceall(i,~isnan(f_C(i,:))),f_C(i,~isnan(f_C(i,:))),2));
    else
        f(i,1) = f_G(i,:);
        f(i,2) = f_V(i,:);
        f(i,3) = f_C(i,:);
    end
    if sum(isnan(f(i,:))) == 0
        rownotnan = [rownotnan;i];
    end
end
f = f(rownotnan,:)/flagdist;
cv = sum(max(g,0),2);
tfe = tfe + size(x,1);
end

function [f_G,f_V,f_C,g,distance] = calculate_objective_PHEV(x,modelnum,batterytype,batteryreplace,cutoff,sequence,maxdist,flag,flagdist,lambda)
% Metamodels
% Coefficients:
% Charge depleting (CD) mode electricity efficiency (eta_E),
% Charge sustain (CS) mode fuel efficiency (eta_G),
% CD and CS mode 0-60 mph acceleration time (t_CD) and (t_CS),
% CD and CS mode battery energy (charging and discharging) per mile (mu_CD) and (mu_CS),
% Final state of charge (SOC) after multiple US06 aggresive driving cycles in CS mode (u_CS) (starting at the target SOC)

eta_E = 0.008*x(:,1).^3 + 0.154*x(:,2).^3 + 0.353*x(:,3).^3 - 0.005*x(:,1).^2.*x(:,2) - 0.005*x(:,1).*x(:,2).^2 - 0.025*x(:,1).^2.*x(:,3) + ...
    0.000*x(:,1).*x(:,3).^2 - 0.057*x(:,2).^2.*x(:,3) - 0.043*x(:,2).*x(:,3).^2 - 0.016*x(:,1).*x(:,2).*x(:,3) - 0.001*x(:,1).^2 - 0.805*x(:,2).^2 - ...
    0.656*x(:,3).^2 + 0.057*x(:,1).*x(:,2) + 0.080*x(:,1).*x(:,3) + 0.342*x(:,2).*x(:,3) - 0.191*x(:,1) + 1.189*x(:,2) - 0.347*x(:,3) + 4.960;

eta_G = 2.214*x(:,1).^3 + 1.087*x(:,2).^3 + 5.578*x(:,3).^3 - 0.815*x(:,1).^2.*x(:,2) + 0.510*x(:,1).*x(:,2).^2 + 1.562*x(:,1).^2.*x(:,3) + ...
    2.212*x(:,1).*x(:,3).^2 - 0.613*x(:,2).^2.*x(:,3) + 0.254*x(:,2).*x(:,3).^2 - 0.159*x(:,1).*x(:,2).*x(:,3) - 8.906*x(:,1).^2 - 6.095*x(:,2).^2 - ...
    15.21*x(:,3).^2 + 0.089*x(:,1).*x(:,2) - 3.274*x(:,1).*x(:,3) + 2.498*x(:,2).*x(:,3) + 2.622*x(:,1) + 9.285*x(:,2) + 5.837*x(:,3) + 57.68;

t_CD = 1.457*x(:,1).^3 - 5.496*x(:,2).^3 - 28.46*x(:,3).^3 + 0.913*x(:,1).^2.*x(:,2) - 0.881*x(:,1).*x(:,2).^2 - 1.050*x(:,1).^2.*x(:,3) - ...
    0.308*x(:,1).*x(:,3).^2 + 2.044*x(:,2).^2.*x(:,3) + 15.61*x(:,2).*x(:,3).^2 + 0.336*x(:,1).*x(:,2).*x(:,3) - 4.634*x(:,1).^2 + 31.48*x(:,2).^2 + ...
    34.02*x(:,3).^2 + 1.153*x(:,1).*x(:,2) + 1.169*x(:,1).*x(:,3) - 32.06*x(:,2).*x(:,3) + 3.405*x(:,1) - 54.47*x(:,2) + 9.570*x(:,3) + 44.23;

t_CS = 3.334*x(:,1).^3 - 2.266*x(:,2).^3 - 20.26*x(:,3).^3 + 0.414*x(:,1).^2.*x(:,2) - 3.524*x(:,1).*x(:,2).^2 - 0.286*x(:,1).^2.*x(:,3) - ...
    10.11*x(:,1).*x(:,3).^2 + 1.951*x(:,2).^2.*x(:,3) + 10.31*x(:,2).*x(:,3).^2 + 5.808*x(:,1).*x(:,2).*x(:,3) - 6.932*x(:,1).^2 + 15.80*x(:,2).^2 + ...
    39.20*x(:,3).^2 + 7.901*x(:,1).*x(:,2) + 6.582*x(:,1).*x(:,3) - 30.12*x(:,2).*x(:,3) - 6.734*x(:,1) - 26.39*x(:,2) - 4.098*x(:,3) + 32.10;

mu_CD = 0.001*x(:,1).^2 + 0.002*x(:,2).^2 + 0.007*x(:,3).^2 + 0.000*x(:,1).*x(:,2) + 0.001*x(:,1).*x(:,3) - 0.001*x(:,2).*x(:,3) + ...
    0.013*x(:,1) + 0.005*x(:,2) + 0.050*x(:,3) + 0.296;

mu_CS = 0.063*x(:,1).^2 - 0.001*x(:,2).^2 - 0.002*x(:,3).^2 - 0.002*x(:,1).*x(:,2) - 0.005*x(:,1).*x(:,3) + 0.001*x(:,2).*x(:,3) - ...
    0.120*x(:,1) + 0.010*x(:,2) + 0.054*x(:,3) + 0.194;

u_CS = -0.194*x(:,1).^2 - 0.005*x(:,2).^2 + 0.047*x(:,3).^2 + 0.000*x(:,1).*x(:,2) + 0.011*x(:,1).*x(:,3) - 0.001*x(:,2).*x(:,3) + ...
    0.382*x(:,1) + 0.019*x(:,2) - 0.077*x(:,3) + 0.140;

% Constraint Functions

g(:,1) = -(11 - t_CD);
g(:,2) = -(11 - t_CS);
g(:,3) = -(u_CS - 0.32);
distance = [];
if flag == 1
    if (sequence == 1 || sequence == 2) && cutoff > 0
        distance = linspace(0,cutoff,ceil(cutoff));
    elseif (sequence == 4 || sequence == 5) && cutoff < maxdist
        distance = linspace(cutoff,maxdist,maxdist+1-ceil(cutoff));
    end
elseif flag == 4
    distance = linspace(0,cutoff,ceil(cutoff)+1);
else
    distance = flagdist;
end

f_G = [];
f_V = [];
f_C = [];
if ~isempty(distance)
    for n = 1:numel(distance)
        s = distance(n);
        % Electric travel and battery degradation
        % Parameters:
        % s_AER = Distance travelled on electricity alone with a fully charged battery
        % s_E and s_G = Distance driven on electric power and gasoline assuming one charge per day
        % kappa = Energy capacity pr battery cell
        kappa = 0.0216;
        s_AER = kappa*1000*x(:,3).*x(:,4).*eta_E;
        s_E = zeros(size(x,1),1);
        s_G = zeros(size(x,1),1);
        for i = 1:size(x,1)
            if s_AER(i) >= s
                s_E(i) = s;
                s_G(i) = 0;
            else
                s_E(i) = s_AER(i);
                s_G(i) = s - s_AER(i);
            end
        end
        
        % Parameters:
        % w_DRV = Daily energy processed while driving
        % w_CHG = Daily energy processed while charging
        % eta_B = Battery charging efficiency
        % r_P = Relative energy capacity decrease
        % r_EOL = Relative energy capacity fade
        % delta = Energy-based depth of discharge (DOD)
        % theta_BAT = Battery life;
        
        % Peterson model
        alpha_DRV = 3.46e-5;
        alpha_CHG = 1.72e-5;
        eta_B = 0.88;
        w_DRV = mu_CD.*s_E + mu_CS.*s_G;
        w_CHG = s_E./(eta_E.*eta_B);
        r_P = (alpha_DRV*w_DRV + alpha_CHG*w_CHG)./(1000*x(:,3)*kappa);
        r_EOL = 1 - x(:,4);
        theta_Peterson = r_EOL./r_P;
        
        % Rosenkranz model
        delta = x(:,4).*s_E./s_AER;
        theta_Rosenkranz = 1441*delta.^(-1.46);
        
        if modelnum == 1
            theta_BAT = theta_Peterson;
        else
            theta_BAT = theta_Rosenkranz;
        end
        
        % Objective Functions
        
        % Obj-1: Petroleum consumption
        % Parameters:
        % f_G = Average gasoline consumption per day
        f_G1 = s_G./eta_G;
        
        % Obj-2: Life cycle greenhouse gas (GHG) emissions
        % Parameters:
        % f_V = Average total life cycle GHG emissions per day
        % v_OP = Operating GHG emissions
        % eta_C = Battery charging efficiency
        % v_E = Electricity emissions
        % v_G = Gasoline life cycle emissions
        % theta_VEH = Vehicle life in days
        % theta_BRPL = Battery replacement effective life
        % v_BAT = Battery pack manufacturing emissions
        % v_B = Life cycle GHG emission associated with battery production
        % v_VEH = Life cycle GHG emission associated with vehicle production
        % s_LIFE = Vehicle life in miles
        eta_C = 0.88;
        v_E = 0.8;
        v_G = 22.8;
        v_OP = (s_E.*v_E)./(eta_E.*eta_C) + (s_G.*v_G)./(eta_G);
        if batterytype == 1
            v_B = 438.44; 
        else
            v_B = 230;
        end
        v_VEH = 6000; 
        s_LIFE = 266640;
        theta_VEH = s_LIFE./s;
        v_BAT = 1000*x(:,3).*kappa.*v_B;
        if batteryreplace == 1
            theta_BRPL = theta_BAT;
        else
            theta_BRPL = min(theta_BAT,theta_VEH);
        end
        f_V1 = v_OP + v_VEH./theta_VEH + v_BAT./theta_BRPL;
        
        % Obj-3: Equivalent annualized cost (EAC)
        % Parameters:
        % N = number of years
        % r_N = Nominal discount rate
        % r_I = Inflation rate
        % r_R = Real discount rate
        % P = Net present value
        % D = Driving days per year
        % T = Vehicle life in years
        % B = Battery life in days
        % c_OP = Sum of cost of electricity needed to charge battery and cost of gasoline consumed
        % c_VEH = Sum of all vehicle related cost
        % c_BASE = Sum of vehicle base cost
        % c_ENG = Engine cost
        % c_MTR = Motor cost
        % c_BAT = Battery pack cost
        % c_B = Battery unit cost
        % c_E = Annual average residential electricity price
        % c_G = Annual average gasoline price
        % f_AP = Capital recovery factor
        % f_C = EAC per driving day
        D = 300;
        c_BASE = 32628.05;
        c_ENG = 17.8*(57*x(:,1)) + 650;
        c_MTR = 26.6*(52*x(:,2)) + 520;
        if batterytype == 1
            c_B = 7073.85;
        else
            c_B = 600;
        end
        c_BAT = 1000*x(:,3)*kappa*c_B;
        c_VEH = c_BASE + c_ENG + c_MTR + c_BAT;
        r_N = 0.028;
        r_I = 0.015;
        c_E = 0.044;
        c_G = 1.288;
        T = theta_VEH./D;
        B = theta_BAT./D;
        c_OP = (s_E.*c_E)./(eta_E.*eta_C) + (s_G.*c_G)./(eta_G);
        r_R = (1 + r_N)./(1 + r_I) - 1;
        f_AP_r_N_T = (r_N*(1 + r_N).^T)./((1 + r_N).^T - 1);
        f_AP_r_R_T = (r_R*(1 + r_R).^T)./((1 + r_R).^T - 1);
        f_AP_r_N_B = (r_N*(1 + r_N).^B)./((1 + r_N).^B - 1);
        if batteryreplace == 1
            f_battery = c_BAT.*f_AP_r_N_B./D;
        else
            f_battery = c_BAT.*f_AP_r_N_T./D;
        end
        f_C1 = c_OP.*f_AP_r_N_T./f_AP_r_R_T + c_VEH.*f_AP_r_N_T./D + f_battery;
        
        % Distribution of vehicle miles traveled per day
        if flag == 1
            f_s = lambda.*exp(-lambda.*s);
        elseif flag == 2
            f_s = 1;
        else
            fun = @(s)lambda.*exp(-lambda.*s);
            f_s = integral(fun,0,maxdist);
        end
        f_G(:,n) = f_G1*f_s;
        f_V(:,n) = f_V1*f_s;
        f_C(:,n) = f_C1*f_s;
    end
end
end

function [f_G,f_V,f_C,g,distance] = calculate_objective_HEV(x,cutoff,sequence,maxdist,flag,flagdist,lambda)

eta_G = 60.1*ones(size(x,1),1);

t_CD = 11*ones(size(x,1),1);

t_CS = 11*ones(size(x,1),1);

u_CS = 0.32*ones(size(x,1),1);

% Constraint Functions

g(:,1) = -(11 - t_CD);
g(:,2) = -(11 - t_CS);
g(:,3) = -(u_CS - 0.32);
distance = [];
if flag == 1
    if (sequence == 3 || sequence == 4) && cutoff > 0
        distance = linspace(0,cutoff,ceil(cutoff));
    elseif (sequence == 1 || sequence == 6) && cutoff < maxdist
        distance = linspace(cutoff,maxdist,maxdist+1-ceil(cutoff));
    end
elseif flag == 4
    distance = linspace(0,cutoff,ceil(cutoff)+1);
else
    distance = flagdist;
end
f_G = [];
f_V = [];
f_C = [];
if ~isempty(distance)
    for n = 1:numel(distance)
        s = distance(n);
        s_G = s*ones(size(x,1),1);
        f_G1 = s_G./eta_G;
        v_G = 11.34;
        v_OP = (s_G.*v_G)./(eta_G);
        v_VEH = 8500;
        s_LIFE = 150000;
        theta_VEH = s_LIFE./s;
        f_V1 = v_OP + v_VEH./theta_VEH;
        D = 300;
        c_B = 600;
        c_BASE = 11183;
        c_ENG = 17.8*57 + 650;
        c_MTR = 26.6*52 + 520;
        c_BAT = 1.3*c_B;
        c_VEH = c_BASE + c_ENG + c_MTR + c_BAT;
        r_N = 0.05;
        r_I = 0.03;
        c_G = 3.30;
        T = theta_VEH./D;
        c_OP = (s_G.*c_G)./(eta_G);
        r_R = (1 + r_N)./(1 + r_I) - 1;
        f_AP_r_N_T = (r_N*(1 + r_N).^T)./((1 + r_N).^T - 1);
        f_AP_r_R_T = (r_R*(1 + r_R).^T)./((1 + r_R).^T - 1);
        f_battery = c_BAT.*f_AP_r_N_T./D;
        f_C1 = c_OP.*f_AP_r_N_T./f_AP_r_R_T + c_VEH.*f_AP_r_N_T./D + f_battery;
        
        % Distribution of vehicle miles traveled per day
        if flag == 1
            f_s = lambda.*exp(-lambda.*s);
        elseif flag == 2
            f_s = 1;
        else
            fun = @(s)lambda.*exp(-lambda.*s);
            f_s = integral(fun,0,maxdist);
        end
        f_G(:,n) = f_G1*f_s;
        f_V(:,n) = f_V1*f_s;
        f_C(:,n) = f_C1*f_s;
    end
end
end

function [f_G,f_V,f_C,g,distance] = calculate_objective_CV(x,cutoff,sequence,maxdist,flag,flagdist,lambda)

eta_G = 29.5*ones(size(x,1),1);

t_CD = 11*ones(size(x,1),1);

t_CS = 11*ones(size(x,1),1);

u_CS = 0.32*ones(size(x,1),1);

% Constraint Functions

g(:,1) = -(11 - t_CD);
g(:,2) = -(11 - t_CS);
g(:,3) = -(u_CS - 0.32);
distance = [];
if flag == 1
    if (sequence == 5 || sequence == 6) && cutoff > 0
        distance = linspace(0,cutoff,ceil(cutoff));
    elseif (sequence == 2 || sequence == 3) && cutoff < maxdist
        distance = linspace(cutoff,maxdist,maxdist+1-ceil(cutoff));
    end
elseif flag == 4
    distance = linspace(0,cutoff,ceil(cutoff)+1);
else
    distance = flagdist;
end
f_G = [];
f_V = [];
f_C = [];
if ~isempty(distance)
    for n = 1:numel(distance)
        s = distance(n);
        s_G = s*ones(size(x,1),1);
        f_G1 = s_G./eta_G;
        v_G = 11.34;
        v_OP = (s_G.*v_G)./(eta_G);
        v_VEH = 8500;
        s_LIFE = 150000;
        theta_VEH = s_LIFE./s;
        f_V1 = v_OP + v_VEH./theta_VEH;
        D = 300;
        c_BASE = 11183;
        c_ENG = 17.8*126 + 650;
        c_VEH = c_BASE + c_ENG;
        r_N = 0.05;
        r_I = 0.03;
        c_G = 3.30;
        T = theta_VEH./D;
        c_OP = (s_G.*c_G)./(eta_G);
        r_R = (1 + r_N)./(1 + r_I) - 1;
        f_AP_r_N_T = (r_N*(1 + r_N).^T)./((1 + r_N).^T - 1);
        f_AP_r_R_T = (r_R*(1 + r_R).^T)./((1 + r_R).^T - 1);
        f_C1 = c_OP.*f_AP_r_N_T./f_AP_r_R_T + c_VEH.*f_AP_r_N_T./D;
        
        % Distribution of vehicle miles traveled per day
        if flag == 1
            f_s = lambda.*exp(-lambda.*s);
        elseif flag == 2
            f_s = 1;
        else
            fun = @(s)lambda.*exp(-lambda.*s);
            f_s = integral(fun,0,maxdist);
        end
        f_G(:,n) = f_G1*f_s;
        f_V(:,n) = f_V1*f_s;
        f_C(:,n) = f_C1*f_s;
    end
end
end