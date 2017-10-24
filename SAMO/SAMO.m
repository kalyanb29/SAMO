function SAMO(param)
tic;
rng(param.seed,'twister');
warning off;
prob=load_problem_definition(param);
archive=[];
param.max_cost = param.pres_func_eval;
param.choose_neighbour = 1;
param.surr_pred_dist = 0.5;
param.surr_tau_err_threshold = 1e-3;
param.choose_surr = 0;
param.surr_eval = 1;
param.local_option = 1;
param.surr_add_crit = 1.e-5;
param.display_gen=0;
% Defining the upper and lower bound  of the variables
limits=zeros(prob.nx,2);
for i=1:prob.nx
    limits(i,:)=prob.range(i,:);
end
LB=limits(:,1)';UB=limits(:,2)';
% Min reference distance computation
if param.surr_eval == 0
    numtotsolutions = param.max_cost;
    ref_sol_set = repmat(LB,numtotsolutions,1)+(repmat(UB,numtotsolutions,1)-repmat(LB,numtotsolutions,1)).*(lhsdesign(numtotsolutions,prob.nx));
    ref_sol_set = (ref_sol_set - repmat(LB,numtotsolutions,1))./(repmat(UB - LB,numtotsolutions,1));
    min_ref_dist = min(pdist(ref_sol_set));
else
    numtotsolutions = param.max_cost;
    ref_sol_set = repmat(LB,numtotsolutions,1)+(repmat(UB,numtotsolutions,1)-repmat(LB,numtotsolutions,1)).*(lhsdesign(numtotsolutions,prob.nx));
    min_ref_dist = 0;
end
% Initializing the population of solutions
X_pop=repmat(LB,param.pop_size,1)+(repmat(UB,param.pop_size,1)-repmat(LB,param.pop_size,1)).*(lhsdesign(param.pop_size,prob.nx));

% Evaluating the initial population of solutions
func=str2func(param.problem_name);
TFE = 0;
[F_pop,G_pop,CVpop,TFE]=func(X_pop,TFE);

% Sorting the initial population
% if strcmp(param.algo,'NSGA-II')
%     [X_pop, F_pop, G_pop]=sort_pop(X_pop, F_pop, G_pop);
% else
%     f_new = new_objective(G_pop, param, prob);
%     [X_pop, F_pop, G_pop] = sort_idea(X_pop, F_pop, G_pop, param, f_new);
% end
% FULL EVALUATION COMPONENT--------------------------------------------------------------------------------
popall = [];
[rankbest]=sort_best(F_pop,CVpop);
if prob.ng > 0
    popall = [popall;zeros(param.pop_size,1) (1:param.pop_size)' X_pop(rankbest,:) F_pop(rankbest,:) G_pop(rankbest,:) CVpop(rankbest,:)];
    archive = [archive; zeros(size(X_pop,1),1) (1:size(X_pop,1))' X_pop F_pop G_pop CVpop];
else
    popall = [popall;zeros(param.pop_size,1) (1:param.pop_size)' X_pop(rankbest,:) F_pop(rankbest,:)];
    archive = [archive; zeros(size(X_pop,1),1) (1:size(X_pop,1))' X_pop F_pop CVpop];
end
fhvall = archive(archive(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
xhvall = archive(archive(:,end) == 0,3:2+prob.nx);
if~isempty(fhvall)
    [front,~] = nd_sort(fhvall,(1:size(fhvall,1))');
    fndall = fhvall(front(1).f,:);
    xndall = xhvall(front(1).f,:);
else
    fndall = [];
    xndall = [];
end
infoall = [xndall fndall];
save(strcat('ndonef_',num2str(TFE),'.mat'),'infoall');
%-----------------------------------------------------------------------------------------------------------------------------
% Initialization of surrogates
tempsurr = Surrogate(param);
cost = TFE;
original_children = [];
for i=1:param.generations
    if TFE >= param.max_cost
        break
    end
    param.num_neighbour = min(size(archive,1),4*round(prob.nx));
    % Doing Binary Tournament, SBX and Polynomial Mutation
    [parent_idx]=min(1:param.pop_size,randperm(param.pop_size));
    [X_childset] = recombination_matrix(LB,UB,param,X_pop(parent_idx,:));
    for j = 1:size(X_childset,1)
        X_child_in = X_childset(j,:);
        % Flag them based on uniqueness == 0 and within max diagonal distance of archive solutions == 1 and outside == 2
        flag(j) = set_flag(archive(:,3:end-1),X_child_in,min_ref_dist,param,prob);
    end
    X_child_surr = X_childset(flag == 1,:);
    X_child_eval = X_childset(flag == 2,:);
    X_child_reject = [];
    X_child_star = [];
    F_child_star = [];
    G_child_star = [];
    Error_star = [];
    Error_reject = [];
    if ~isempty(X_child_surr)
        for j = 1:size(X_child_surr,1)
            X_child_in = X_child_surr(j,:);
            [id_neighbour,rangesurr] = choose_neighbour(archive(:,3:end-1),X_child_in,param,prob);
            % Assign blank surr
            surr_init = tempsurr;
            % Set range for surrogates;
            surrset = set_range(surr_init, rangesurr);
            % Add data to surrogate model
            surradd = add_pop(surrset, prob, archive(id_neighbour,3:end-1));
            % Train the surrogate based on archive
            surr = trainsurr(surradd, param, prob);
            datamodelsetup = surr.model_data;
            original_children = [original_children;i X_child_in datamodelsetup.allerror];
            if datamodelsetup.error <= param.surr_tau_err_threshold || param.local_option == 1
                [X_loc,F_loc,G_loc] = run_subea(X_child_in,surr,param,prob,rangesurr);
                X_child_star = [X_child_star;X_loc];
                F_child_star = [F_child_star;F_loc];
                G_child_star = [G_child_star;G_loc];
                Error_star = [Error_star;datamodelsetup.error];
            else
                X_child_reject = [X_child_reject;X_child_in];
                Error_reject = [Error_reject;datamodelsetup.error];
            end
        end
    end
    X_child_all = X_child_eval;
    if ~isempty(X_child_reject)
        [~,idminerror] = min(Error_reject);
        X_child_all = [X_child_all;X_child_reject(idminerror,:)];
        %         X_child = [X_child_eval;X_child_reject];
    end
    if ~isempty(X_child_star)
        if ~isempty(G_child_star)
            CV_child_star = sum(max(G_child_star,0),2);
        else
            CV_child_star = zeros(size(X_child_star,1),1);
        end
        rank_child_star = sort_best(F_child_star,CV_child_star);
        X_child_all = [X_child_all;X_child_star(rank_child_star,:)];
    end
    X_child_all1 = round(X_child_all*1e4)/1e4;
    [~,idunique] = unique(X_child_all1,'rows','stable');
    X_child_allu = X_child_all(idunique,:);
    numchildx = min(param.pop_size,size(X_child_allu,1));
    child_x = X_child_allu(1:numchildx,:);
    costleft = param.max_cost - TFE;
    if size(child_x,1) >= costleft
        numflag = costleft;
    else
        numflag = size(child_x,1);
    end
    if numflag == 0
        break
    end
    X_child = child_x(1:numflag,:);
    % FULL EVALUATION POLICY--------------------------------------------------------------------------------------------------
    % Evaluating the children
    if ~isempty(X_child)
        [F_child,G_child,~,TFE]=func(X_child,TFE);
    else
        F_child = [];
        G_child = [];
    end
    numcommonarchive = sum(ismember(X_child,archive(:,3:2+prob.nx),'rows','legacy'));
    TFE = TFE - numcommonarchive;
    % Sorting the parent and child solutions to identify parents for next gen
    X_pop=[X_pop;X_child];F_pop=[F_pop;F_child];G_pop=[G_pop;G_child];
    [X_pop, F_pop, G_pop]=sort_pop(X_pop, F_pop, G_pop);
    X_pop=X_pop(1:param.pop_size,:);
    F_pop=F_pop(1:param.pop_size,:);
    if(~isempty(G_pop))
        G_pop=G_pop(1:param.pop_size,:);
        CVpop = sum(max(G_pop,0),2);
    else
        G_pop=[];
        CVpop = zeros(param.pop_size,1);
    end
    if ~isempty(G_child)
        CVchild = sum(max(G_child,0),2);
    else
        CVchild = zeros(size(X_child,1),1);
    end
    [rankbest]=sort_best(F_pop,CVpop);
    if prob.ng > 0
        popall = [popall;i*ones(param.pop_size,1) (1:param.pop_size)' X_pop(rankbest,:) F_pop(rankbest,:) G_pop(rankbest,:) CVpop(rankbest,:)];
        archive = [archive; zeros(size(X_child,1),1) (1:size(X_child,1))' X_child F_child G_child CVchild];
        
    else
        popall = [popall;i*ones(param.pop_size,1) (1:param.pop_size)' X_pop(rankbest,:) F_pop(rankbest,:)];
        archive = [archive; zeros(size(X_child,1),1) (1:size(X_child,1))' X_child F_child CVchild];
    end
    fhvall = archive(archive(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
    xhvall = archive(archive(:,end) == 0,3:2+prob.nx);
    if~isempty(fhvall)
        [front,~] = nd_sort(fhvall,(1:size(fhvall,1))');
        fndall = fhvall(front(1).f,:);
        xndall = xhvall(front(1).f,:);
    else
        fndall = [];
        xndall = [];
    end
    infoall = [xndall fndall];
    save(strcat('ndonef_',num2str(TFE),'.mat'),'infoall');
    if(param.display_gen==1)
        if(size(F_pop,2)==1)
            if  CVpop(rankbest(1)) == 0
                plot(TFE,F_pop(rankbest(1)),'b*');hold on;
                plot(TFE,CVpop(rankbest(1)),'b+');hold on;
            else
                plot(TFE,F_pop(rankbest(1)),'r*');hold on;
                plot(TFE,CVpop(rankbest(1)),'r+');hold on;
            end
        end
        if(size(F_pop,2)==2)
            plot(F_pop(CVpop == 0,1),F_pop(CVpop == 0,2),'b.');hold on;
            plot(F_pop(CVpop > 0,1),F_pop(CVpop > 0,2),'r*');hold on;
            hold off;
        end
        if(size(F_pop,2)==3)
            plot3(F_pop(CVpop == 0,1),F_pop(CVpop == 0,2),F_pop(CVpop == 0,3),'b.');hold on;
            plot3(F_pop(CVpop > 0,1),F_pop(CVpop > 0,2),F_pop(CVpop > 0,3),'r*');hold on;
            hold off;
        end
    end
    cost = [cost;TFE];
end
fhvall = archive(archive(:,end) == 0,3+prob.nx:2+prob.nx+prob.nf);
xhvall = archive(archive(:,end) == 0,3:2+prob.nx);
if~isempty(fhvall)
    [front,~] = nd_sort(fhvall,(1:size(fhvall,1))');
    fndall = fhvall(front(1).f,:);
    xndall = xhvall(front(1).f,:);
else
    fndall = [];
    xndall = [];
end
infoall = [xndall fndall];
save(strcat('ndonef_',num2str(TFE),'.mat'),'infoall');
%-----------------------------------------------------------------------
timeexec = toc;
save(strcat(param.problem_name,'_runtime.mat'),'timeexec');
feaspop = F_pop(CVpop == 0,:);
[~,idx] = nd_rank_one(feaspop,1);
ndfeas = feaspop(idx,:);
save(strcat(param.problem_name,'_original_childrenstat.mat'),'original_children');
save(strcat(param.problem_name,'_cost.mat'),'cost');
save(strcat(param.problem_name,'_ndfeasibleobj.mat'),'ndfeas');
save(strcat(param.problem_name,'_Archive.mat'),'archive');
save(strcat(param.problem_name,'_All.mat'),'popall');
end

function [prob]=load_problem_definition(def)
funh=str2func(def.problem_name);
prob=funh();
end

function [rank]=sort_best(f,cv)
id = (1:numel(cv))';
j=1;l=1;Infeas=[];feasible=[];
for i=1:size(id,1)
    if(cv(id(i)) == 0)
        feasible(l)=id(i);
        l=l+1;
    else
        Infeas(j)=id(i);
        j=j+1;
    end
end

if(~isempty(feasible))
    if(size(f,2)==1)
        [~,I]=sort(f(feasible)); % for single objective
        feasible=feasible(I);
    else
        idfeas = (1:numel(feasible))';
        [~,I] = nd_sort(f(feasible,:),idfeas);
        feasible=feasible(I);
    end
end
if(~isempty(Infeas))
    vec=cv(Infeas);
    [~,I]=sort(vec);
    Infeas=Infeas(I);
end
rank=[feasible Infeas]';
end

function [fronts,idx] = nd_sort(f_all, id)
idx = [];
if isempty(f_all)
    fronts = [];
    return
end

if nargin == 1
    id = (1:size(f_all,1))';
end

if isempty(id)
    fronts = [];
    return
end

try
    fronts = nd_sort_c(id, f_all(id,:));
catch
    warning('ND_SORT() MEX not available. Using slower matlab version.');
    fronts = nd_sort_m(id, f_all(id,:));
end
for i = 1:size(fronts,2)
    [ranks, dist] = sort_crowding(f_all, fronts(i).f);
    idx = [idx;ranks];
end

end

function [F] = nd_sort_c(feasible, f_all)
[frontS, frontS_n] = ind_sort2(feasible', f_all');
F = [];
for i = 1:length(feasible)
    count = frontS_n(i);
    if count > 0
        tmp = frontS(1:count, i) + 1;
        F(i).f = feasible(tmp)';
    end
end
end

function [F] = nd_sort_m(feasible, f_all)

front = 1;
F(front).f = [];

N = length(feasible);
M = size(f_all,2);

individual = [];
for i = 1:N
    id1 = feasible(i);
    individual(id1).N = 0;
    individual(id1).S = [];
end

% Assignging dominate flags
for i = 1:N
    id1 = feasible(i);
    f = repmat(f_all(i,:), N, 1);
    dom_less = sum(f <= f_all, 2);
    dom_more = sum(f >= f_all, 2);
    for j = 1:N
        id2 = feasible(j);
        if dom_less(j) == M && dom_more(j) < M
            individual(id1).S = [individual(id1).S id2];
        elseif dom_more(j) == M && dom_less(j) < M
            individual(id1).N = individual(id1).N + 1;
        end
    end
end

% identifying the first front
for i = 1:N
    id1 = feasible(i);
    if individual(id1).N == 0
        F(front).f = [F(front).f id1];
    end
end

% Identifying the rest of the fronts
while ~isempty(F(front).f)
    H = [];
    for i = 1 : length(F(front).f)
        p = F(front).f(i);
        if ~isempty(individual(p).S)
            for j = 1 : length(individual(p).S)
                q = individual(p).S(j);
                individual(q).N = individual(q).N - 1;
                if individual(q).N == 0
                    H = [H q];
                end
            end
        end
    end
    if ~isempty(H)
        front = front+1;
        F(front).f = H;
    else
        break
    end
end
end

function [ranks, dist] = sort_crowding(f_all, front_f)

L = length(front_f);
if L == 1
    ranks = front_f;
    dist = Inf;
else
    dist = zeros(L, 1);
    nf = size(f_all, 2);
    
    for i = 1:nf
        f = f_all(front_f, i);		% get ith objective
        [tmp, I] = sort(f);
        scale = f(I(L)) - f(I(1));
        dist(I(1)) = Inf;
        for j = 2:L-1
            id = I(j);
            id1 = front_f(I(j-1));
            id2 = front_f(I(j+1));
            if scale > 0
                dist(id) = dist(id) + (f_all(id2,i)-f_all(id1,i)) / scale;
            end
        end
    end
    dist = dist / nf;
    [tmp, I] = sort(dist, 'descend');
    ranks = front_f(I)';
end
end

function [ranks] = sort_constr(cv, id)
[tmp, I] = sort(cv(id));
ranks = id(I);
end

function  [X_pop, F_pop, G_pop]=sort_pop(X_pop, F_pop, G_pop)
N=size(X_pop,1);
G1_pop = G_pop;
if(~isempty(G_pop))
    G1_pop(G1_pop <0) = 0;
    cvsum = sum(G1_pop,2);
    feasible=find(cvsum == 0);
    infeasible=setdiff((1:N)',feasible);
else
    feasible=(1:N)';
    infeasible=[];
    cvsum=zeros(N,1);
end

% For the feasible solutions
ranks1 = [];
if ~isempty(feasible)
    feasiblepop = F_pop(feasible,:);
    [~,id] = nd_sort(feasiblepop,(1:numel(feasible))');
    appended_list = feasible(id);
    ranks1 = appended_list;
end

% For the infeasible solutions
ranks2 = [];
if ~isempty(infeasible)
    ranks2 = sort_constr(cvsum, infeasible);
end
assert(length(ranks1) + length(ranks2)==N);
pop_rank = [ranks1 ; ranks2];
X_pop=X_pop(pop_rank(1:N),:);
F_pop=F_pop(pop_rank(1:N),:);

if(~isempty(G_pop))
    G_pop=G_pop(pop_rank(1:N),:);
else
    G_pop=[];
end
end

function [fout] = value2quantile(f,option,range)
if option == 1 % 1 = normalize;
    if nargin == 3
        fout = (f - repmat(range(1,:),size(f,1),1))./repmat((range(2,:)-range(1,:)),size(f,1),1);
    else
        fout = (f - repmat(min(f,[],1),size(f,1),1))./repmat((max(f,[],1)-min(f,[],1)),size(f,1),1);
    end
else % 0 = denormalize
    if nargin == 3
        fout = repmat(range(1,:),size(f,1),1) + f.*repmat((range(2,:)-range(1,:)),size(f,1),1);
    else
        fout = repmat(min(f),size(f,1),1) + f.*repmat((max(f)-min(f)),size(f,1),1);
    end
end
end

function [id_use,range] = choose_neighbour(archive,x,param,prob)
xpop = archive(:,1:prob.nx);
diffmat = zeros(size(xpop,1),1);
for j = 1:size(xpop,1)
    diffmat(j) = sqrt(sum((x - xpop(j,:)).^2,2));
end
numinvalid = 0;
rownaninf = double.empty(0,1);
for i = 1:size(archive,1)
    if sum(isinf(archive(i,:)) > 0) || sum(isnan(archive(i,:)) > 0)
        numinvalid = numinvalid + 1;
        rownaninf = [rownaninf;i];
    end
end
if param.choose_neighbour == 0
    id_use = find(diffmat <= (min(diffmat)+(max(diffmat)-min(diffmat))*param.radius));
else
    [~,id_sort] = sort(diffmat);
    id_num = id_sort(1:param.num_neighbour);
    idtotuse = union(id_num,rownaninf,'rows','stable');
    idlefttot = setdiff(id_sort,idtotuse,'rows','stable');
    idremain = setdiff(id_num,rownaninf,'rows','stable');
    if numel(idremain) < prob.nx + 1
        addelem = prob.nx + 1 - numel(idremain);
        id_num = [id_num;idlefttot(1:min(addelem,numel(idlefttot)))];
    end
    id_use = id_num;
end

rangemin = min(xpop(id_use,:),[],1)';
rangemax = max(xpop(id_use,:),[],1)';
range = [rangemin rangemax];
end

function [ndset,idx] = nd_rank_one(set1,dir)
if nargin == 1
    dir = 1;
end
[N,M] = size(set1);
switch(dir)
    case 1
        dom = nd_sort_min(set1, M, N);
    case 2
        dom = nd_sort_max(set1, M, N);
    otherwise
        error('wrong value of dir');
end

idx = [];
for i = 1:N
    if dom(i) == 0
        idx = [idx i];
    end
end
ndset = set1(idx(:),:);
end

function [dom] = nd_sort_min(set1, M, N)
dom = zeros(1, N);
for i = 1:N
    f = repmat(set1(i,:), N, 1);
    dom_less = sum(f <= set1, 2);
    dom_more = sum(f >= set1, 2);
    for j = 1:N
        if dom_less(j) == M && dom_more(j) < M
            dom(j) = dom(j) + 1;
        end
    end
end
end

function [dom] = nd_sort_max(set1, M, N)
dom = zeros(1, N);
for i = 1:N
    f = repmat(set1(i,:), N, 1);
    dom_less = sum(f <= set1, 2);
    dom_more = sum(f >= set1, 2);
    for j = 1:N
        if dom_more(j) == M && dom_less(j) < M
            dom(j) = dom(j) + 1;
        end
    end
end
end

function [y1,y2] = pred_constr(x,surr,prob)
datamodelsetup = surr.model_data;
datamodel = datamodelsetup.csdata{1}.model;
typemodel = datamodelsetup.csdata{1}.type;
typeset = typemodel(prob.nf+1:end);
modelset = datamodel(prob.nf+1:end);
y1 = zeros(1,length(modelset));
x = normalize(surr, x);
for i = 1:length(typeset)
    type = typeset{i};
    model = modelset{i};
    ptype = type;
    pred_func = strcat(ptype, '_predict');
    y1(:,i) = feval(pred_func, x, model);
end
y2 = [];
end

function [y] = pred_obj(x,surr,prob)
datamodelsetup = surr.model_data;
datamodel = datamodelsetup.csdata{1}.model;
typemodel = datamodelsetup.csdata{1}.type;
typeset = typemodel(1:prob.nf);
modelset = datamodel(1:prob.nf);
y = zeros(1,length(modelset));
x = normalize(surr, x);
for i = 1:length(typeset)
    type = typeset{i};
    model = modelset{i};
    ptype = type;
    pred_func = strcat(ptype, '_predict');
    y(:,i) = feval(pred_func, x, model);
end
end

function [X_child] = recombination_matrix(LB,UB,param,X_parent_ordered)
[X_child] = crossover_SBX_matrix(LB,UB,param,X_parent_ordered);
[X_child] = mutation_POLY_matrix(LB,UB,param,X_child);
end

function [y1, y2] = op_SBX_matrix(l_limit,u_limit, x1, x2, eta)
y1 = x1;
y2 = x2;
ipos = abs(x1-x2) > 1e-6;
if sum(ipos) >0
    x1_op = x1(ipos);
    x2_op = x2(ipos);
    l_op = l_limit(ipos);
    u_op = u_limit(ipos);
    pos_swap = x2_op < x1_op;
    tmp = x1_op(pos_swap);
    x1_op(pos_swap) = x2_op(pos_swap);
    x2_op(pos_swap) = tmp;
    r = rand(size(x1_op));
    beta = 1 + (2*(x1_op - l_op)./(x2_op - x1_op));
    alpha = 2 - beta.^-(eta+1);
    betaq = (1./(2-r.*alpha)).^(1/(eta+1));
    betaq(r <= 1./alpha) = (r(r <= 1./alpha).*alpha(r <= 1./alpha)).^(1/(eta+1));
    y1_op = 0.5 * (x1_op + x2_op - betaq.*(x2_op - x1_op));
    
    beta = 1 + 2*(u_op - x2_op)./(x2_op - x1_op);
    alpha = 2 - beta.^-(eta+1);
    betaq = (1./(2-r.*alpha)).^(1/(eta+1));
    betaq(r <= 1./alpha) = (r(r <= 1./alpha).*alpha(r <= 1./alpha)).^(1/(eta+1));
    y2_op = 0.5 * (x1_op + x2_op + betaq.*(x2_op - x1_op));
    
    y1_op(y1_op < l_op) = l_op(y1_op < l_op);
    y1_op(y1_op > u_op) = u_op(y1_op > u_op);
    
    y2_op(y2_op < l_op) = l_op(y2_op < l_op);
    y2_op(y2_op > u_op) = u_op(y2_op > u_op);
    
    pos_swap = (rand(size(x1_op,1),1) <= 0.5);
    tmp = y1_op(pos_swap);
    y1_op(pos_swap) = y2_op(pos_swap);
    y2_op(pos_swap) = tmp;
    
    y1(ipos) = y1_op;
    y2(ipos) = y2_op;
end
end

function [c, fn_evals] = crossover_SBX_matrix(LB,UB,param,p)

c = p; % parent size = no_solution*no_variable.
fn_evals = 0;
A = rand(floor(size(p,1)/2),1);
is_crossover =[(A <= param.crossover_prob)';(A <= param.crossover_prob)'];
p_cross = p(is_crossover,:);
[N,m] = size(p_cross);
c_cross = p_cross;
p1_cross = p_cross(1:2:N,:);
p2_cross = p_cross(2:2:N,:);
B = rand(size(p_cross,1)/2,length(LB));
l_limit = repmat(LB,size(p_cross,1)/2,1);
u_limit = repmat(UB,size(p_cross,1)/2,1);
cross_pos = (B <= 0.5);
l_cross = l_limit(cross_pos);
u_cross = u_limit(cross_pos);
p1 = p1_cross(cross_pos);
p2 = p2_cross(cross_pos);
c1 = p1_cross;
c2 = p2_cross;
if size(l_cross,1) ~= size(p1,1)
    p1 = p1';
    p2 = p2';
end
[y1, y2] = op_SBX_matrix(l_cross,u_cross,p1,p2,param.crossover_sbx_eta);
c1(cross_pos) = y1;
c2(cross_pos) = y2;
c_cross(1:2:N,:) = c1;
c_cross(2:2:N,:) = c2;
c(is_crossover,:) = c_cross;

if(mod(p,2)~=0)
    p1=p(end,:);
    p2=p(randi(size(p,1)),:);
    for i=1:size(p1,2)
        temp=p1;
        if(rand<=param.crossover_prob)
            [temp(i),~]=SBX(p1(i),p2(i),LB(i),UB(i),param.crossover_sbx_eta);
        end
    end
    c(end,:)=temp;
end

end

function [y1, y2] = SBX(x1, x2, x_min, x_max, eta)
% Make sure the variables are not the same
if abs(x1-x2) > 1.e-6
    % make sure x1 < x2
    if x2 < x1, tmp = x1; x1 = x2; x2 = tmp; end
    
    r = rand(1);
    beta = 1 + (2 * (x1-x_min) / (x2-x1));
    alpha = 2 - beta^-(eta+1);
    if r <= 1/alpha
        betaq = (r*alpha)^(1/(eta+1));
    else
        betaq = (1/(2-r*alpha))^(1/(eta+1));
    end
    y1 = 0.5 * (x1+x2 - betaq*(x2-x1));
    
    beta = 1 + (2 * (x_max-x2) / (x2-x1));
    alpha = 2 - beta^-(eta+1);
    if r <= 1/alpha
        betaq = (r*alpha)^(1/(eta+1));
    else
        betaq = (1/(2-r*alpha))^(1/(eta+1));
    end
    y2 = 0.5 * (x1+x2 + betaq*(x2-x1));
    
    if y1 < x_min, y1 = x_min; end
    if y1 > x_max, y1 = x_max; end
    
    if y2 < x_min, y2 = x_min; end
    if y2 > x_max, y2 = x_max; end
    
    if rand(1) <= 0.5, tmp = y1; y1 = y2; y2 = tmp; end
else
    y1 = x1;
    y2 = x2;
end
end

function [x] = op_POLY_matrix(l_limit,u_limit,p,eta)
x = p;
x_min = l_limit((u_limit - l_limit) > 0);
x_max = u_limit((u_limit - l_limit) > 0);
x_mut = p((u_limit - l_limit) > 0);
delta1 = (x_mut - x_min)./(x_max - x_min);
delta2 = (x_max - x_mut)./(x_max - x_min);
mut_pow = 1/(eta+1);
r = rand(size(x_min));
xy = 1 - delta2;
val = 2*(1-r) + 2*(r-0.5) .* xy.^(eta+1);
deltaq = 1 - val.^mut_pow;
xy(r <= 0.5) = 1 - delta1(r <= 0.5);
val(r <= 0.5) = 2*r(r <= 0.5) + (1-2*r(r <= 0.5)).* xy(r <= 0.5).^(eta+1);
deltaq(r <= 0.5) = val(r <= 0.5).^mut_pow - 1;
temp = x_mut + deltaq.*(x_max-x_min);

temp(temp < x_min) = x_min(temp < x_min);
temp(temp > x_max) = x_max(temp > x_max);
x((u_limit - l_limit) > 0) = temp;
end

function [p, fn_evals] = mutation_POLY_matrix(LB,UB,param, p)
fn_evals = 0;
A = rand(size(p,1),length(UB));
l_limit = repmat(LB,size(p,1),1);
u_limit = repmat(UB,size(p,1),1);
p_mut = p(A <= param.mutation_prob);
l_mut = l_limit(A <= param.mutation_prob);
u_mut = u_limit(A <= param.mutation_prob);
p_mut = op_POLY_matrix(l_mut,u_mut,p_mut,param.mutation_poly_eta);
p(A <= param.mutation_prob) = p_mut;
end

function[X_loc,F_loc,G_loc] = run_subea(x,surr,param,prob,rangesurr)
% Initialization
xpop = repmat(rangesurr(:,1)',param.subea_pop_size-1,1)+(repmat(rangesurr(:,2)',param.subea_pop_size-1,1)-repmat(rangesurr(:,1)',param.subea_pop_size-1,1)).*(lhsdesign(param.subea_pop_size-1,prob.nx));
X_pop = [x;xpop];
for i = 1:size(X_pop,1)
    F_pop(i,:) = pred_obj(X_pop(i,:),surr,prob);
    if prob.ng > 0
        [G_pop(i,:),~] = pred_constr(X_pop(i,:),surr,prob);
    else
        G_pop = [];
    end
end
% Sort initialization
[X_pop, F_pop, G_pop]=sort_pop(X_pop, F_pop, G_pop);
for i = 1:param.subea_generations
    [parent_idx]=min(1:param.subea_pop_size,randperm(param.subea_pop_size));
    [X_child] = recombination_matrix(rangesurr(:,1)',rangesurr(:,2)',param,X_pop(parent_idx,:));
    for j = 1:size(X_pop,1)
        F_child(j,:) = pred_obj(X_child(j,:),surr,prob);
        if prob.ng > 0
            [G_child(j,:),~] = pred_constr(X_child(j,:),surr,prob);
        else
            G_child = [];
        end
    end
    X_pop=[X_pop;X_child];F_pop=[F_pop;F_child];G_pop=[G_pop;G_child];
    [X_pop, F_pop, G_pop]=sort_pop(X_pop, F_pop, G_pop);
    X_pop=X_pop(1:param.subea_pop_size,:);
    F_pop=F_pop(1:param.subea_pop_size,:);
    if(~isempty(G_pop))
        G_pop=G_pop(1:param.subea_pop_size,:);
        CVpop = sum(max(G_pop,0),2);
    else
        G_pop=[];
        CVpop = zeros(param.subea_pop_size,1);
    end
end
[rankbest]=sort_best(F_pop,CVpop);
X_loc = X_pop(rankbest,:);
F_loc = F_pop(rankbest,:);
if ~isempty(G_pop)
    G_loc = G_pop(rankbest,:);
else
    G_loc = [];
end
end

function[flag] = set_flag(archive,x_old,min_ref_dist,param,prob)
func = str2func(param.problem_name);
dummy = 0;
[ftest,gtest,~,~] = func(x_old,dummy);
x_pop = archive(:,1:prob.nx);
diffmat = zeros(size(x_pop,1),1);
range = prob.range;
x = (x_old - repmat(range(:,1)',size(x_old,1),1))./(repmat(range(:,2)' - range(:,1)',size(x_old,1),1));
xpop = (x_pop - repmat(range(:,1)',size(x_pop,1),1))./(repmat(range(:,2)' - range(:,1)',size(x_pop,1),1));
for j = 1:size(xpop,1)
    diffmat(j) = sqrt(sum((x - xpop(j,:)).^2,2));
end
if param.choose_neighbour == 0
    id_use = find(diffmat <= (min(diffmat)+(max(diffmat)-min(diffmat))*param.radius));
else
    [~,id_sort] = sort(diffmat);
    id_use = id_sort(1:param.num_neighbour);
end
mindiffmat = min(diffmat(id_use));
xpop1 = round(xpop*1e4)/1e4;
x1 = round(x*1e4)/1e4;
existance = sum(ismember(xpop1,x1,'rows'));
if existance > 0 || isinf(sum(ftest)) || isinf(sum(gtest)) || isnan(sum(ftest)) || isnan(sum(gtest))
    flag = 0;
elseif mindiffmat > min_ref_dist && param.surr_eval == 0
    flag = 2;
else
    flag = 1;
end
end

function [cdata] = Cluster(data, seed, method, varargin)

% Save random state
rand_state = RandStream.getGlobalStream.State;

% Start new random
rng(seed,'twister');

cdata.centroid = [];
cdata.id = {};

switch method
    case 'k_means'
        if nargin ~= 4
            error('Fourth argument (number of clusters) missing');
        end
        k = varargin{1};
        warning('off', 'all');
        [IDX, C] = kmeans(data, k, 'EmptyAction', 'singleton', ...
            'Replicates', 5, 'Display', 'off');
        warning('on', 'all');
        cdata.centroid = C;
        for i = 1:k
            cdata.id{i} = find(IDX == i);
        end
    case 'k_medoids'
        if nargin ~= 4
            error('Fourth argument (number of clusters) missing');
        end
        k = varargin{1};
        warning('off', 'all');
        [IDX, C] = kmeans(data, k, 'EmptyAction', 'singleton', ...
            'Replicates', 5, 'Display', 'off');
        warning('on', 'all');
        for i = 1:k
            id = find(IDX == i);
            cdata.id{i} = id;
            tmp = data(id,:) - repmat(C(i,:), length(id), 1);
            dist = sum(tmp .* tmp, 2);
            [tmp, I] = min(dist);
            cdata.centroid(i) = id(I);
        end
    otherwise
        error('Unknown clustering method');
end

% cdata = class(cdata, 'Cluster');

% Restore random state
RandStream.getGlobalStream.State = rand_state;
end

function close(logger)
fclose(logger.fp);
end

function [logger] = Logger(prefix, batch_mode, varargin)

logger.filename = '';
logger.batch_mode = 0;
logger.fp = [];

logger = class(logger, 'Logger');

if nargin >= 2
    logger.filename = sprintf('%s.log', prefix);
    logger.batch_mode = batch_mode;
    
    append_flag = 0;
    if nargin == 3
        append_flag = varargin{1};
    end
    
    if append_flag == 0
        logger = open(logger);
    else
        logger = reopn(logger);
    end
end
end

function [logger] = open(logger)
logger.fp = fopen(logger.filename, 'w');
end

function [logger] = reopen(logger)
logger.fp = fopen(logger.filename, 'a+');
end

function [logger] = savelog(logger)
close(logger);
logger = reopen(logger);
end

function write(logger, varargin)
if ~isempty(logger.fp)
    fprintf(logger.fp, varargin{:});
end
if logger.batch_mode == 0 || logger.batch_mode == 1
    %fprintf(varargin{:});
end
end

function [val] = getval(range)
val=range.val;
end

function [obj] = Range(type, val)

if nargin == 2
    obj.type = lower(type);
    switch obj.type
        case 'scalar'
            obj.val = val(1);
        case 'range'
            obj.val= val(1:2);
        case 'irange'
            obj.val = val(1:2);
        case {'set', 'subset'}
            assert(iscell(val));
            if ~iscellstr(val)
                val = cell2mat(val);
            end
            obj.val = sort(val);
        otherwise
            error('Unknown range type. Use scalar/range/irange/set/subset.');
    end
else
    obj.type = 'scalar';
    obj.val = [];
end

obj = class(obj, 'Range');
end

function [x] = sample(range, N)

if nargin == 1
    N = 1;
end

switch range.type
    case 'scalar'
        x = ones(N,1) * range.val;
    case 'range'
        x = range.val(1) + (range.val(2)-range.val(1)) * rand(N,1);
    case 'irange'
        x = randint(N,1,range.val);
    case 'set'
        x = range.val(randint(N,1,[1, length(range.val)]));
end
end

function  [r, dr] = corrcubic(theta, d)
%CORRCUBIC  Cubic correlation function,
%
%           n
%   r_i = prod max(0, 1 - 3(theta_j*d_ij)^2 + 2(theta_j*d_ij)^3) ,  i = 1,...,m
%          j=1
%
% If length(theta) = 1, then the model is isotropic:
% all theta_j = theta.
%
% Call:    r = corrcubic(theta, d)
%          [r, dr] = corrcubic(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n  matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update June 25, 2002

[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
    theta = repmat(theta,1,n);
elseif  length(theta) ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
else
    theta = theta(:).';
end
td = min(abs(d) .* repmat(theta,m,1), 1);
ss = 1 - td.^2 .* (3 - 2*td);
r = prod(ss, 2);

if  nargout > 1
    dr = zeros(m,n);
    for  j = 1 : n
        dd = 6*theta(j) * sign(d(:,j)) .* td(:,j) .* (td(:,j) - 1);
        dr(:,j) = prod(ss(:,[1:j-1 j+1:n]),2) .* dd;
    end
end
end

function  [r, dr] = correxp(theta, d)
%CORREXP  Exponential correlation function
%
%           n
%   r_i = prod exp(-theta_j * |d_ij|)
%          j=1
%
% If length(theta) = 1, then the model is isotropic:
% theta_j = theta(1), j=1,...,n
%
% Call:    r = correxp(theta, d)
%          [r, dr] = correxp(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update April 12, 2002

[m n] = size(d);  % number of differences and dimension of data
lt = length(theta);
if  lt == 1,  theta = repmat(theta,1,n);
elseif  lt ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
else
    theta = theta(:).';
end

td = abs(d) .* repmat(-theta, m, 1);
r = exp(sum(td,2));

if  nargout > 1
    dr = repmat(-theta,m,1) .* sign(d) .* repmat(r,1,n);
end
end

function  [r, dr] = correxpg(theta, d)
%CORREXPG  General exponential correlation function
%
%           n
%   r_i = prod exp(-theta_j * d_ij^theta_n+1)
%          j=1
%
% If n > 1 and length(theta) = 2, then the model is isotropic:
% theta_j = theta(1), j=1,...,n;  theta_(n+1) = theta(2)
%
% Call:    r = correxpg(theta, d)
%          [r, dr] = correxpg(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update April 12, 2002

[m n] = size(d);  % number of differences and dimension of data
lt = length(theta);
if  n > 1 & lt == 2
    theta = [repmat(theta(1),1,n)  theta(2)];
elseif  lt ~= n+1
    error(sprintf('Length of theta must be 2 or %d',n+1))
else
    theta = theta(:).';
end

pow = theta(end);   tt = repmat(-theta(1:n), m, 1);
td = abs(d).^pow .* tt;
r = exp(sum(td,2));

if  nargout > 1
    dr = pow  * tt .* sign(d) .* (abs(d) .^ (pow-1)) .* repmat(r,1,n);
end
end

function  [r, dr] = corrgauss(theta, d)
%CORRGAUSS  Gaussian correlation function,
%
%           n
%   r_i = prod exp(-theta_j * d_ij^2) ,  i = 1,...,m
%          j=1
%
% If length(theta) = 1, then the model is isotropic:
% all  theta_j = theta .
%
% Call:    r = corrgauss(theta, d)
%          [r, dr] = corrgauss(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update June 2, 2002

[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
    theta = repmat(theta,1,n);
elseif  length(theta) ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
end

td = d.^2 .* repmat(-theta(:).',m,1);
r = exp(sum(td, 2));

if  nargout > 1
    dr = repmat(-2*theta(:).',m,1) .* d .* repmat(r,1,n);
end
end

function  [r, dr] = corrlin(theta, d)
%CORRLIN  Linear correlation function,
%
%           n
%   r_i = prod max(0, 1 - theta_j * d_ij) ,  i = 1,...,m
%          j=1
%
% If length(theta) = 1, then the model is isotropic:
% all theta_j = theta .
%
% Call:    r = corrlin(theta, d)
%          [r, dr] = corrlin(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update April 12, 2002

[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
    theta = repmat(theta,1,n);
elseif  length(theta) ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
end

td = max(1 - abs(d) .* repmat(theta(:).',m,1), 0);
r = prod(td, 2);

if  nargout > 1
    dr = zeros(m,n);
    for  j = 1 : n
        dr(:,j) = prod(td(:,[1:j-1 j+1:n]),2) .* (-theta(j) * sign(d(:,j)));
    end
end
end

function  [r, dr] = corrspherical(theta, d)
%CORRSPHERICAL  Spherical correlation function,
%
%           n
%   r_i = prod max(0, 1 - 1.5(theta_j*d_ij) + .5(theta_j*d_ij)^3) ,  i = 1,...,m
%          j=1
%
% If length(theta) = 1, then the model is isotropic:
% all  theta_j = theta .
%
% Call:    r = corrspherical(theta, d)
%          [r, dr] = corrspherical(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update April 12, 2002

[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
    theta = repmat(theta,1,n);
elseif  length(theta) ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
else
    theta = theta(:).';
end
td = min(abs(d) .* repmat(theta,m,1), 1);
ss = 1 - td .* (1.5 - .5*td.^2);
r = prod(ss, 2);

if  nargout > 1
    dr = zeros(m,n);
    for  j = 1 : n
        dd = 1.5*theta(j) * sign(d(:,j)).*(td(:,j).^2 - 1);
        dr(:,j) = prod(ss(:,[1:j-1 j+1:n]),2) .* dd;
    end
end
end

function  [r, dr] = corrspline(theta, d)
%CORRSPLINE  Cubic spline correlation function,
%
%           n
%   r_i = prod S(theta_j*d_ij) ,  i = 1,...,m
%          j=1
%
% with
%           1 - 15x^2 + 30x^3   for   0 <= x <= 0.5
%   S(x) =  1.25(1 - x)^3       for  0.5 < x < 1
%           0                   for    x >= 1
% If length(theta) = 1, then the model is isotropic:
% all  theta_j = theta.
%
% Call:    r = corrspline(theta, d)
%          [r, dr] = corrspline(theta, d)
%
% theta :  parameters in the correlation function
% d     :  m*n matrix with differences between given data points
% r     :  correlation
% dr    :  m*n matrix with the Jacobian of r at x. It is
%          assumed that x is given implicitly by d(i,:) = x - S(i,:),
%          where S(i,:) is the i'th design site.

% hbn@imm.dtu.dk
% Last update May 30, 2002

[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
    theta = repmat(theta,1,n);
elseif  length(theta) ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
else
    theta = theta(:).';
end
mn = m*n;   ss = zeros(mn,1);
xi = reshape(abs(d) .* repmat(theta,m,1), mn,1);
% Contributions to first and second part of spline
i1 = find(xi <= 0.2);
i2 = find(0.2 < xi & xi < 1);
if  ~isempty(i1)
    ss(i1) = 1 - xi(i1).^2 .* (15  - 30*xi(i1));
end
if  ~isempty(i2)
    ss(i2) = 1.25 * (1 - xi(i2)).^3;
end
% Values of correlation
ss = reshape(ss,m,n);
r = prod(ss, 2);

if  nargout > 1  % get Jacobian
    u = reshape(sign(d) .* repmat(theta,m,1), mn,1);
    dr = zeros(mn,1);
    if  ~isempty(i1)
        dr(i1) = u(i1) .* ( (90*xi(i1) - 30) .* xi(i1) );
    end
    if  ~isempty(i2)
        dr(i2) = -3.75 * u(i2) .* (1 - xi(i2)).^2;
    end
    ii = 1 : m;
    for  j = 1 : n
        sj = ss(:,j);  ss(:,j) = dr(ii);
        dr(ii) = prod(ss,2);
        ss(:,j) = sj;   ii = ii + m;
    end
    dr = reshape(dr,m,n);
end  % Jacobian
end

function  [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
%DACEFIT Constrained non-linear least-squares fit of a given correlation
% model to the provided data set and regression model
%
% Call
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0)
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
%
% Input
% S, Y    : Data points (S(i,:), Y(i,:)), i = 1,...,m
% regr    : Function handle to a regression model
% corr    : Function handle to a correlation function
% theta0  : Initial guess on theta, the correlation function parameters
% lob,upb : If present, then lower and upper bounds on theta
%           Otherwise, theta0 is used for theta
%
% Output
% dmodel  : DACE model: a struct with the elements
%    regr   : function handle to the regression model
%    corr   : function handle to the correlation function
%    theta  : correlation function parameters
%    beta   : generalized least squares estimate
%    gamma  : correlation factors
%    sigma2 : maximum likelihood estimate of the process variance
%    S      : scaled design sites
%    Ssc    : scaling factors for design arguments
%    Ysc    : scaling factors for design ordinates
%    C      : Cholesky factor of correlation matrix
%    Ft     : Decorrelated regression matrix
%    G      : From QR factorization: Ft = Q*G' .
% perf    : struct with performance information. Elements
%    nv     : Number of evaluations of objective function
%    perf   : (q+2)*nv array, where q is the number of elements
%             in theta, and the columns hold current values of
%                 [theta;  psi(theta);  type]
%             |type| = 1, 2 or 3, indicate 'start', 'explore' or 'move'
%             A negative value for type indicates an uphill step

% hbn@imm.dtu.dk
% Last update September 3, 2002

% Check design points
[m n] = size(S);  % number of design sites and their dimension
sY = size(Y);
if  min(sY) == 1,  Y = Y(:);   lY = max(sY);  sY = size(Y);
else,              lY = sY(1); end
if m ~= lY
    error('S and Y must have the same number of rows'), end

% Check correlation parameters
lth = length(theta0);
if  nargin > 5  % optimization case
    if  length(lob) ~= lth | length(upb) ~= lth
        error('theta0, lob and upb must have the same length'), end
    if  any(lob <= 0) | any(upb < lob)
        error('The bounds must satisfy  0 < lob <= upb'), end
else  % given theta
    if  any(theta0 <= 0)
        error('theta0 must be strictly positive'), end
end

% Normalize data
mS = mean(S);   sS = std(S);
mY = mean(Y);   sY = std(Y);
% 02.08.27: Check for 'missing dimension'
j = find(sS == 0);
if  ~isempty(j),  sS(j) = 1; end
j = find(sY == 0);
if  ~isempty(j),  sY(j) = 1; end
S = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);
Y = (Y - repmat(mY,m,1)) ./ repmat(sY,m,1);

% Calculate distances D between points
mzmax = m*(m-1) / 2;        % number of non-zero distances
ij = zeros(mzmax, 2);       % initialize matrix with indices
D = zeros(mzmax, n);        % initialize matrix with distances
ll = 0;
for k = 1 : m-1
    ll = ll(end) + (1 : m-k);
    ij(ll,:) = [repmat(k, m-k, 1) (k+1 : m)']; % indices for sparse matrix
    D(ll,:) = repmat(S(k,:), m-k, 1) - S(k+1:m,:); % differences between points
end
if  min(sum(abs(D),2) ) == 0
    error('Multiple design sites are not allowed'), end

% Regression matrix
F = feval(regr, S);  [mF p] = size(F);
if  mF ~= m, error('number of rows in  F  and  S  do not match'), end
if  p > mF,  error('least squares problem is underdetermined'), end

% parameters for objective function
par = struct('corr',corr, 'regr',regr, 'y',Y, 'F',F, ...
    'D', D, 'ij',ij, 'scS',sS);

% Determine theta
if  nargin > 5
    % Bound constrained non-linear optimization
    [theta f fit perf] = boxmin(theta0, lob, upb, par);
    if  isinf(f)
        error('Bad parameter region.  Try increasing  upb'), end
else
    % Given theta
    theta = theta0(:);
    [f  fit] = objfunc(theta, par);
    perf = struct('perf',[theta; f; 1], 'nv',1);
    if  isinf(f)
        error('Bad point.  Try increasing theta0'), end
end

% Return values
dmodel = struct('regr',regr, 'corr',corr, 'theta',theta.', ...
    'beta',fit.beta, 'gamma',fit.gamma, 'sigma2',sY.^2.*fit.sigma2, ...
    'S',S, 'Ssc',[mS; sS], 'Ysc',[mY; sY], ...
    'C',fit.C, 'Ft',fit.Ft, 'G',fit.G);
end
% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================

function  [obj, fit] = objfunc(theta, par)
% Initialize
obj = inf;
fit = struct('sigma2',NaN, 'beta',NaN, 'gamma',NaN, ...
    'C',NaN, 'Ft',NaN, 'G',NaN);
m = size(par.F,1);
% Set up  R
r = feval(par.corr, theta, par.D);
idx = find(r > 0);   o = (1 : m)';
mu = (10+m)*eps;
R = sparse([par.ij(idx,1); o], [par.ij(idx,2); o], ...
    [r(idx); ones(m,1)+mu]);
% Cholesky factorization with check for pos. def.
[C rd] = chol(R);
if  rd,  return, end % not positive definite

% Get least squares solution
C = C';   Ft = C \ par.F;
[Q G] = qr(Ft,0);
if  rcond(G) < 1e-10
    % Check   F
    if  cond(par.F) > 1e15
        T = sprintf('F is too ill conditioned\nPoor combination of regression model and design sites');
        error(T)
    else  % Matrix  Ft  is too ill conditioned
        return
    end
end
Yt = C \ par.y;   beta = G \ (Q'*Yt);
rho = Yt - Ft*beta;  sigma2 = sum(rho.^2)/m;
detR = prod( full(diag(C)) .^ (2/m) );
obj = sum(sigma2) * detR;
if  nargout > 1
    fit = struct('sigma2',sigma2, 'beta',beta, 'gamma',rho' / C, ...
        'C',C, 'Ft',Ft, 'G',G');
end
end
% --------------------------------------------------------

function  [t, f, fit, perf] = boxmin(t0, lo, up, par)
%BOXMIN  Minimize with positive box constraints

% Initialize
[t, f, fit, itpar] = start(t0, lo, up, par);
if  ~isinf(f)
    % Iterate
    p = length(t);
    if  p <= 2,  kmax = 2; else,  kmax = min(p,4); end
    for  k = 1 : kmax
        th = t;
        [t, f, fit, itpar] = explore(t, f, fit, itpar, par);
        [t, f, fit, itpar] = move(th, t, f, fit, itpar, par);
    end
end
perf = struct('nv',itpar.nv, 'perf',itpar.perf(:,1:itpar.nv));
end
% --------------------------------------------------------

function  [t, f, fit, itpar] = start(t0, lo, up, par)
% Get starting point and iteration parameters

% Initialize
t = t0(:);  lo = lo(:);   up = up(:);   p = length(t);
D = 2 .^ ([1:p]'/(p+2));
ee = find(up == lo);  % Equality constraints
if  ~isempty(ee)
    D(ee) = ones(length(ee),1);   t(ee) = up(ee);
end
ng = find(t < lo | up < t);  % Free starting values
if  ~isempty(ng)
    t(ng) = (lo(ng) .* up(ng).^7).^(1/8);  % Starting point
end
ne = find(D ~= 1);

% Check starting point and initialize performance info
[f  fit] = objfunc(t,par);   nv = 1;
itpar = struct('D',D, 'ne',ne, 'lo',lo, 'up',up, ...
    'perf',zeros(p+2,200*p), 'nv',1);
itpar.perf(:,1) = [t; f; 1];
if  isinf(f)    % Bad parameter region
    return
end

if  length(ng) > 1  % Try to improve starting guess
    d0 = 16;  d1 = 2;   q = length(ng);
    th = t;   fh = f;   jdom = ng(1);
    for  k = 1 : q
        j = ng(k);    fk = fh;  tk = th;
        DD = ones(p,1);  DD(ng) = repmat(1/d1,q,1);  DD(j) = 1/d0;
        alpha = min(log(lo(ng) ./ th(ng)) ./ log(DD(ng))) / 5;
        v = DD .^ alpha;   tk = th;
        for  rept = 1 : 4
            tt = tk .* v;
            [ff  fitt] = objfunc(tt,par);  nv = nv+1;
            itpar.perf(:,nv) = [tt; ff; 1];
            if  ff <= fk
                tk = tt;  fk = ff;
                if  ff <= f
                    t = tt;  f = ff;  fit = fitt; jdom = j;
                end
            else
                itpar.perf(end,nv) = -1;   break
            end
        end
    end % improve
    
    % Update Delta
    if  jdom > 1
        D([1 jdom]) = D([jdom 1]);
        itpar.D = D;
    end
end % free variables

itpar.nv = nv;
end
% --------------------------------------------------------

function  [t, f, fit, itpar] = explore(t, f, fit, itpar, par)
% Explore step

nv = itpar.nv;   ne = itpar.ne;
for  k = 1 : length(ne)
    j = ne(k);   tt = t;   DD = itpar.D(j);
    if  t(j) == itpar.up(j)
        atbd = 1;   tt(j) = t(j) / sqrt(DD);
    elseif  t(j) == itpar.lo(j)
        atbd = 1;  tt(j) = t(j) * sqrt(DD);
    else
        atbd = 0;  tt(j) = min(itpar.up(j), t(j)*DD);
    end
    [ff  fitt] = objfunc(tt,par);  nv = nv+1;
    itpar.perf(:,nv) = [tt; ff; 2];
    if  ff < f
        t = tt;  f = ff;  fit = fitt;
    else
        itpar.perf(end,nv) = -2;
        if  ~atbd  % try decrease
            tt(j) = max(itpar.lo(j), t(j)/DD);
            [ff  fitt] = objfunc(tt,par);  nv = nv+1;
            itpar.perf(:,nv) = [tt; ff; 2];
            if  ff < f
                t = tt;  f = ff;  fit = fitt;
            else
                itpar.perf(end,nv) = -2;
            end
        end
    end
end % k

itpar.nv = nv;
end

% --------------------------------------------------------

function  [t, f, fit, itpar] = move(th, t, f, fit, itpar, par)
% Pattern move

nv = itpar.nv;   ne = itpar.ne;   p = length(t);
v = t ./ th;
if  all(v == 1)
    itpar.D = itpar.D([2:p 1]).^.2;
    return
end

% Proper move
rept = 1;
while  rept
    tt = min(itpar.up, max(itpar.lo, t .* v));
    [ff  fitt] = objfunc(tt,par);  nv = nv+1;
    itpar.perf(:,nv) = [tt; ff; 3];
    if  ff < f
        t = tt;  f = ff;  fit = fitt;
        v = v .^ 2;
    else
        itpar.perf(end,nv) = -3;
        rept = 0;
    end
    if  any(tt == itpar.lo | tt == itpar.up), rept = 0; end
end

itpar.nv = nv;
itpar.D = itpar.D([2:p 1]).^.25;
end

function  [mS, mY] = dsmerge(S, Y, ds, nms, wtds, wtdy)
%DSMERGE  Merge data for multiple design sites.
%
% Call
%   [mS, mY] = dsmerge(S, Y)
%   [mS, mY] = dsmerge(S, Y, ds)
%   [mS, mY] = dsmerge(S, Y, ds, nms)
%   [mS, mY] = dsmerge(S, Y, ds, nms, wtds)
%   [mS, mY] = dsmerge(S, Y, ds, nms, wtds, wtdy)
%
% Input
% S, Y : Data points (S(i,:), Y(i,:)), i = 1,...,m
% ds   : Threshold for equal, normalized sites. Default is 1e-14.
% nms  : Norm, in which the distance is measured.
%        nms =  1 : 1-norm (sum of absolute coordinate differences)
%               2 : 2-norm (Euclidean distance) (default)
%        otherwise: infinity norm (max coordinate difference)
% wtds : What to do with the S-values in case of multiple points.
%        wtds = 1 : return the mean value (default)
%               2 : return the median value
%               3 : return the 'cluster center'
% wtdy : What to do with the Y-values in case of multiple points.
%        wtdy = 1 : return the mean value (default)
%               2 : return the median value
%               3 : return the 'cluster center' value
%               4 : return the minimum value
%               5 : return the maximum value
%
% Output
% mS : Compressed design sites, with multiple points merged
%      according to wtds
% mY : Responses, compressed according to wtdy

% hbn@imm.dtu.dk
% Last update July 3, 2002

% Check design points
[m n] = size(S);  % number of design sites and their dimension
sY = size(Y);
if  min(sY) == 1,  Y = Y(:);   lY = max(sY);  sY = size(Y);
else,              lY = sY(1); end
if m ~= lY
    error('S and Y must have the same number of rows'), end

% Threshold
if  nargin < 3
    ds = 1e-14;
elseif  (ds < 0) | (ds > .5)
    error('ds must be in the range [0, 0.5]'), end

% Which measure
if  nargin < 4
    nms = 2;
elseif  (nms ~= 1) & (nms ~= 2)
    nms = Inf;
end

% What to do
if  nargin < 5
    wtds = 1;
else
    wtds = round(wtds);
    if  (wtds < 1) | (wtds > 3)
        error('wtds must be in the range [1, 3]'), end
end
if  nargin < 6
    wtdy = 1;
else
    wtdy = round(wtdy);
    if  (wtdy < 1) | (wtdy > 5)
        error('wtdy must be in the range [1, 5]'), end
end

% Process data
more = 1;
ladr = zeros(1,ceil(m/2));
while more
    m = size(S,1);
    D = zeros(m,m);
    
    % Normalize sites
    mS = mean(S);   sS = std(S);
    scS = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);
    
    % Calculate distances D (upper triangle of the symetric matrix)
    for k = 1 : m-1
        kk = k+1 : m;
        dk = abs(repmat(scS(k,:), m-k, 1) - scS(kk,:));
        if  nms == 1,      D(kk,k) = sum(dk,2);
        elseif  nms == 2,  D(kk,k) = sqrt(sum(dk.^2,2));
        else,              D(kk,k) = max(dk,[],2);      end
    end
    D = D + D'; % make D symetric
    
    % Check distances
    mult = zeros(1,m);
    for  j = 1 : m
        % Find the number of multiple sites in each column of D
        mult(j) = length(find(D(:,j) < ds));
    end
    % Find the first column with the maximum number of multiple sites
    [mmult jj] = max(mult);
    
    if  mmult == 1,  more = 0;
    else
        nm = 0;
        while  mmult > 1
            nm = nm + 1;  % no. of points to merge
            ladr(nm) = jj;
            
            % Merge point no jj and its neighbours, note that jj is the center
            % of the cluster, as it has the most neighbors (among the multiple sites)
            ngb = find(D(:,jj) < ds);
            
            switch  wtds
                case 1,  S(jj,:) = mean(S(ngb,:));
                case 2,  S(jj,:) = median(S(ngb,:));
                case 3,  S(jj,:) = S(jj,:);
            end
            
            switch  wtdy
                case 1,  Y(jj,:) = mean(Y(ngb,:));
                case 2,  Y(jj,:) = median(Y(ngb,:));
                case 3,  Y(jj,:) = Y(jj,:);
                case 4,  Y(jj,:) = min(Y(ngb,:));
                case 5,  Y(jj,:) = max(Y(ngb,:));
            end
            
            % Delete from list
            mult(ngb) = 0;
            [mmult jj] = max(mult);
        end
        
        % Reduced data set
        act = [find(mult > 0)  ladr(1:nm)];
        S = S(act,:);    Y = Y(act,:);
    end % multiple
end % loop

% Return reduced set
mS = S;   mY = Y;
end

function  S = gridsamp(range, q)
%GRIDSAMP  n-dimensional grid over given range
%
% Call:    S = gridsamp(range, q)
%
% range :  2*n matrix with lower and upper limits
% q     :  n-vector, q(j) is the number of points
%          in the j'th direction.
%          If q is a scalar, then all q(j) = q
% S     :  m*n array with points, m = prod(q)

% hbn@imm.dtu.dk
% Last update June 25, 2002

[mr n] = size(range);    dr = diff(range);
if  mr ~= 2 | any(dr < 0)
    error('range must be an array with two rows and range(1,:) <= range(2,:)')
end
sq = size(q);
if  min(sq) > 1 | any(q <= 0)
    error('q must be a vector with non-negative elements')
end
p = length(q);
if  p == 1,  q = repmat(q,1,n);
elseif  p ~= n
    error(sprintf('length of q must be either 1 or %d',n))
end

% Check for degenerate intervals
i = find(dr == 0);
if  ~isempty(i),  q(i) = 0*q(i); end

% Recursive computation
if  n > 1
    A = gridsamp(range(:,2:end), q(2:end));  % Recursive call
    [m p] = size(A);   q = q(1);
    S = [zeros(m*q,1) repmat(A,q,1)];
    y = linspace(range(1,1),range(2,1), q);
    k = 1:m;
    for  i = 1 : q
        S(k,1) = repmat(y(i),m,1);  k = k + m;
    end
else
    S = linspace(range(1,1),range(2,1), q).';
end
end

function S = lhsamp(m, n)
%LHSAMP  Latin hypercube distributed random numbers
%
% Call:    S = lhsamp
%          S = lhsamp(m)
%          S = lhsamp(m, n)
%
% m : number of sample points to generate, if unspecified m = 1
% n : number of dimensions, if unspecified n = m
%
% S : the generated n dimensional m sample points chosen from
%     uniform distributions on m subdivions of the interval (0.0, 1.0)

% hbn@imm.dtu.dk
% Last update April 12, 2002

if nargin < 1, m = 1; end
if nargin < 2, n = m; end

S = zeros(m,n);
for i = 1 : n
    S(:, i) = (rand(1, m) + (randperm(m) - 1))' / m;
end
end

function  [y, or1, or2, dmse] = predictor(x, dmodel)
%PREDICTOR  Predictor for y(x) using the given DACE model.
%
% Call:   y = predictor(x, dmodel)
%         [y, or] = predictor(x, dmodel)
%         [y, dy, mse] = predictor(x, dmodel)
%         [y, dy, mse, dmse] = predictor(x, dmodel)
%
% Input
% x      : trial design sites with n dimensions.
%          For mx trial sites x:
%          If mx = 1, then both a row and a column vector is accepted,
%          otherwise, x must be an mx*n matrix with the sites stored
%          rowwise.
% dmodel : Struct with DACE model; see DACEFIT
%
% Output
% y    : predicted response at x.
% or   : If mx = 1, then or = gradient vector/Jacobian matrix of predictor
%        otherwise, or is an vector with mx rows containing the estimated
%                   mean squared error of the predictor
% Three or four results are allowed only when mx = 1,
% dy   : Gradient of predictor; column vector with  n elements
% mse  : Estimated mean squared error of the predictor;
% dmse : Gradient vector/Jacobian matrix of mse

% hbn@imm.dtu.dk
% Last update August 26, 2002

or1 = NaN;   or2 = NaN;  dmse = NaN;  % Default return values
if  isnan(dmodel.beta)
    y = NaN;
    error('DMODEL has not been found')
end

[m n] = size(dmodel.S);  % number of design sites and number of dimensions
sx = size(x);            % number of trial sites and their dimension
if  min(sx) == 1 & n > 1 % Single trial point
    nx = max(sx);
    if  nx == n
        mx = 1;  x = x(:).';
    end
else
    mx = sx(1);  nx = sx(2);
end
if  nx ~= n
    error(sprintf('Dimension of trial sites should be %d',n))
end

% Normalize trial sites
x = (x - repmat(dmodel.Ssc(1,:),mx,1)) ./ repmat(dmodel.Ssc(2,:),mx,1);
q = size(dmodel.Ysc,2);  % number of response functions
y = zeros(mx,q);         % initialize result

if  mx == 1  % one site only
    dx = repmat(x,m,1) - dmodel.S;  % distances to design sites
    if  nargout > 1                 % gradient/Jacobian wanted
        [f df] = feval(dmodel.regr, x);
        [r dr] = feval(dmodel.corr, dmodel.theta, dx);
        % Scaled Jacobian
        dy = (df * dmodel.beta).' + dmodel.gamma * dr;
        % Unscaled Jacobian
        or1 = dy .* repmat(dmodel.Ysc(2, :)', 1, nx) ./ repmat(dmodel.Ssc(2,:), q, 1);
        if q == 1
            % Gradient as a column vector
            or1 = or1';
        end
        if  nargout > 2  % MSE wanted
            
            rt = dmodel.C \ r;
            u = dmodel.Ft.' * rt - f.';
            v = dmodel.G \ u;
            or2 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + sum(v.^2) - sum(rt.^2))',1,q);
            
            if  nargout > 3  % gradient/Jacobian of MSE wanted
                % Scaled gradient as a row vector
                Gv = dmodel.G' \ v;
                g = (dmodel.Ft * Gv - rt)' * (dmodel.C \ dr) - (df * Gv)';
                % Unscaled Jacobian
                dmse = repmat(2 * dmodel.sigma2',1,nx) .* repmat(g ./ dmodel.Ssc(2,:),q,1);
                if q == 1
                    % Gradient as a column vector
                    dmse = dmse';
                end
            end
            
        end
        
    else  % predictor only
        f = feval(dmodel.regr, x);
        r = feval(dmodel.corr, dmodel.theta, dx);
    end
    
    % Scaled predictor
    sy = f * dmodel.beta + (dmodel.gamma*r).';
    % Predictor
    y = (dmodel.Ysc(1,:) + dmodel.Ysc(2,:) .* sy)';
    
else  % several trial sites
    % Get distances to design sites
    dx = zeros(mx*m,n);  kk = 1:m;
    for  k = 1 : mx
        dx(kk,:) = repmat(x(k,:),m,1) - dmodel.S;
        kk = kk + m;
    end
    % Get regression function and correlation
    f = feval(dmodel.regr, x);
    r = feval(dmodel.corr, dmodel.theta, dx);
    r = reshape(r, m, mx);
    
    % Scaled predictor
    sy = f * dmodel.beta + (dmodel.gamma * r).';
    % Predictor
    y = repmat(dmodel.Ysc(1,:),mx,1) + repmat(dmodel.Ysc(2,:),mx,1) .* sy;
    
    if  nargout > 1   % MSE wanted
        rt = dmodel.C \ r;
        u = dmodel.G \ (dmodel.Ft.' * rt - f.');
        or1 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + colsum(u.^2) - colsum(rt.^2))',1,q);
        if  nargout > 2
            disp('WARNING from PREDICTOR.  Only  y  and  or1=mse  are computed')
        end
    end
    
end % of several sites
end
% >>>>>>>>>>>>>>>>   Auxiliary function  ====================

function  s = colsum(x)
% Columnwise sum of elements in  x
if  size(x,1) == 1,  s = x;
else,                s = sum(x);  end
end

function options = rbfcreate(x, y, varargin)
%RBFCREATE Creates an RBF interpolation
%   OPTIONS = RBFSET(X, Y, 'NAME1',VALUE1,'NAME2',VALUE2,...) creates an
%   radial base function interpolation
%
%   RBFCREATE with no input arguments displays all property names and their
%   possible values.
%
%RBFCREATE PROPERTIES
%

%
% Alex Chirokov, alex.chirokov@gmail.com
% 16 Feb 2006
tic;
% Print out possible values of properties.
if (nargin == 0) & (nargout == 0)
    fprintf('               x: [ dim by n matrix of coordinates for the nodes ]\n');
    fprintf('               y: [   1 by n vector of values at nodes ]\n');
    fprintf('     RBFFunction: [ gaussian  | thinplate | cubic | multiquadrics | {linear} ]\n');
    fprintf('     RBFConstant: [ positive scalar     ]\n');
    fprintf('       RBFSmooth: [ positive scalar {0} ]\n');
    fprintf('           Stats: [ on | {off} ]\n');
    fprintf('\n');
    return;
end
Names = [
    'RBFFunction      '
    'RBFConstant      '
    'RBFSmooth        '
    'Stats            '
    ];
[m,n] = size(Names);
names = lower(Names);

options = [];
for j = 1:m
    options.(deblank(Names(j,:))) = [];
end

%**************************************************************************
%Check input arrays
%**************************************************************************
[nXDim nXCount]=size(x);
[nYDim nYCount]=size(y);

if (nXCount~=nYCount)
    error(sprintf('x and y should have the same number of rows'));
end;

if (nYDim~=1)
    error(sprintf('y should be n by 1 vector'));
end;

options.('x')           = x;
options.('y')           = y;
%**************************************************************************
%Default values
%**************************************************************************
options.('RBFFunction') = 'linear';
options.('RBFConstant') = (prod(max(x')-min(x'))/nXCount)^(1/nXDim); %approx. average distance between the nodes
options.('RBFSmooth')   = 0;
options.('Stats')       = 'off';

%**************************************************************************
% Argument parsing code: similar to ODESET.m
%**************************************************************************

i = 1;
% A finite state machine to parse name-value pairs.
if rem(nargin-2,2) ~= 0
    error('Arguments must occur in name-value pairs.');
end
expectval = 0;                          % start expecting a name, not a value
while i <= nargin-2
    arg = varargin{i};
    
    if ~expectval
        if ~isstr(arg)
            error(sprintf('Expected argument %d to be a string property name.', i));
        end
        
        lowArg = lower(arg);
        j = strmatch(lowArg,names);
        if isempty(j)                       % if no matches
            error(sprintf('Unrecognized property name ''%s''.', arg));
        elseif length(j) > 1                % if more than one match
            % Check for any exact matches (in case any names are subsets of others)
            k = strmatch(lowArg,names,'exact');
            if length(k) == 1
                j = k;
            else
                msg = sprintf('Ambiguous property name ''%s'' ', arg);
                msg = [msg '(' deblank(Names(j(1),:))];
                for k = j(2:length(j))'
                    msg = [msg ', ' deblank(Names(k,:))];
                end
                msg = sprintf('%s).', msg);
                error(msg);
            end
        end
        expectval = 1;                      % we expect a value next
        
    else
        options.(deblank(Names(j,:))) = arg;
        expectval = 0;
    end
    i = i + 1;
end

if expectval
    error(sprintf('Expected value for property ''%s''.', arg));
end


%**************************************************************************
% Creating RBF Interpolatin
%**************************************************************************

switch lower(options.('RBFFunction'))
    case 'linear'
        options.('rbfphi')   = @rbfphi_linear;
    case 'cubic'
        options.('rbfphi')   = @rbfphi_cubic;
    case 'multiquadric'
        options.('rbfphi')   = @rbfphi_multiquadrics;
    case 'thinplate'
        options.('rbfphi')   = @rbfphi_thinplate;
    case 'gaussian'
        options.('rbfphi')   = @rbfphi_gaussian;
    otherwise
        options.('rbfphi')   = @rbfphi_linear;
end

phi       = options.('rbfphi');

A=rbfAssemble(x, phi, options.('RBFConstant'), options.('RBFSmooth'));

b=[y'; zeros(nXDim+1, 1)];

%inverse
rbfcoeff=A\b;

%SVD
% [U,S,V] = svd(A);
%
% for i=1:1:nXCount+1
%     if (S(i,i)>0) S(i,i)=1/S(i,i); end;
% end;
% rbfcoeff = V*S'*U*b;


options.('rbfcoeff') = rbfcoeff;


if (strcmp(options.('Stats'),'on'))
    fprintf('%d point RBF interpolation was created in %e sec\n', length(y), toc);
    fprintf('\n');
end;
end
function [A]=rbfAssemble(x, phi, const, smooth)
[dim n]=size(x);
A=zeros(n,n);
for i=1:n
    for j=1:i
        r=norm(x(:,i)-x(:,j));
        temp=feval(phi,r, const);
        A(i,j)=temp;
        A(j,i)=temp;
    end
    A(i,i) = A(i,i) - smooth;
end
% Polynomial part
P=[ones(n,1) x'];
A = [ A      P
    P' zeros(dim+1,dim+1)];
end
%**************************************************************************
% Radial Base Functions
%**************************************************************************
function u=rbfphi_linear(r, const)
u=r;
end

function u=rbfphi_cubic(r, const)
u=r.*r.*r;
end

function u=rbfphi_gaussian(r, const)
u=exp(-r.*r/(const*const));
end

function u=rbfphi_multiquadrics(r, const)
u=sqrt(1+r.*r/(const*const));
end

function u=rbfphi_thinplate(r, const)
u=r.*r.*log(r+1);
end

function [f] = rbfinterp(x, options)
tic;
phi       = options.('rbfphi');
rbfconst  = options.('RBFConstant');
nodes     = options.('x');
rbfcoeff  = (options.('rbfcoeff'))';


[dim              n] = size(nodes);
[dimPoints  nPoints] = size(x);

if (dim~=dimPoints)
    error(sprintf('x should have the same number of rows as an array used to create RBF interpolation'));
end;

f = zeros(1, nPoints);
r = zeros(1, n);

for i=1:1:nPoints
    s=0;
    for j=1:n
        r(j) =  norm(x(:,i) - nodes(:,j));
    end
    
    s = rbfcoeff(n+1) + sum(rbfcoeff(1:n).*feval(phi, r, rbfconst));
    %s = rbfcoeff(n+1) + sum(rbfcoeff(1:n).*rbfphi_multiquadrics( r, rbfconst));
    
    for k=1:dim
        s=s+rbfcoeff(k+n+1)*x(k,i);     % linear part
    end
    f(i) = s;
end;

if (strcmp(options.('Stats'),'on'))
    fprintf('Interpolation at %d points was computed in %e sec\n', length(f), toc);
end;
end

function writesurr(param,surr, varargin)
if ~isempty(surr.logger)
    write(surr.logger, varargin{:});
else
    if param.display_gen == 1
        fprintf(varargin{:});
    end
end
end

function y = adaptive_mlp_predict(X, mmodel)
% ADAPTIVE_MLP_PREDICT - To predict using MLP model
%
% Call
%    y = mlp_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : MLP model obtained using mlp_model
%
% Output:
% y   : Predicted response
%

% Check arguments
if nargin ~= 2
    error('mlp_predict requires 2 input arguments')
end

y = sim(mmodel, X')';
end

function mmodel = adaptive_mlp_train(X, Y)
% ADAPTIVE_MLP_MODEL - To construct Adpative MLP model
%
% Call
%    rmodel = mlp_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
%
% Output
% rmodel : MLP Model
%

% Check arguments
if nargin ~= 2
    error('mlp_model requires 2 input arguments')
end

% Check design points
[m1, nx] = size(X);
[m2, ny] = size(Y);
assert(m1 == m2, 'X and Y must have the same number of rows.');

ni = round(nx/2) : max(nx,4);

actfunc = {'tansig' 'purelin'};
tmp_model = cell(1, length(ni));
perf = zeros(1, length(ni));
for i = 1:length(ni);
    neuron = [i];
    tmp_model{i} = newff(X', Y', neuron, actfunc);
    tmp_model{i}.trainParam.epochs = 5000;
    tmp_model{i}.trainParam.showWindow = false;
    [tmp_model{i}, tr] = train(tmp_model{i}, X', Y');
    perf(i) = tr.perf(tr.epoch == tr.best_epoch);
end

[tmp, I] = min(perf);
mmodel = tmp_model{I};
end

function [surr] = add_points(surr, x, y)
% ADD_POINTS() adds observations (decision+response) for the surrogate model
%

if isempty(surr.range)
    error('Range of decision variables not defined.');
end

assert(size(x,1) == size(y,1));
assert(size(x,2) == surr.nx);

if surr.ny == 0
    surr.ny = size(y,2);
else
    assert(size(y,2) == surr.ny);
end

N = size(x,1);

% Verify that the new point is not in the neighborhood of archived points
for i = 1:N
    % ignore points with objectives/constraints set to inf
    x_normal = normalize(surr, x(i,:));
    is_new = 1;
    if surr.count > 0
        for j = 1:surr.count
            if norm(surr.x_normal(j,:) - x_normal, 2) < surr.add_crit
                is_new = 0;
                break
            end
        end
    end
    if is_new == 1
        surr.x(surr.count+1,:) = x(i,:);
        surr.x_normal(surr.count+1,:) = x_normal;
        surr.y(surr.count+1,:) = y(i,:);
        surr.count = surr.count + 1;
    end
end
end

function [surr] = add_pop(surr, prob, pop)
% ADD_POP() adds points from the population to the surrogate
xpop = pop(:,1:prob.nx);
fpop = pop(:,prob.nx+1:prob.nx+prob.nf);
if prob.ng > 0
    gpop = pop(:,prob.nx+prob.nf+1:prob.nx+prob.nf+prob.ng);
else
    gpop = [];
end
% Need to add only solutions that are evaluated truly
x = xpop;
y = [fpop gpop];
surr = add_points(surr, x, y);
end

function y = dace_predict(X, dmodel)
% DACEPREDICT - To predict using kriging model
%
% Call
%    y = dace_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : Kriging model obtained using dace_model
%
% Output:
% y   : Predicted response
%

% Check arguments
if nargin ~= 2
    error('dace_predict requires 2 input arguments')
end

f = predictor(X, dmodel);
if size(f,1) == size(X,1)
    y = f;
else
    y = f';
end
end

function dmodel = dace_train(X, Y)
% DACEMODEL - To construct kriging model
%
% Call
%    rmodel = dace_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
%
% Output
% dmodel : DACE Model
%

% Check arguments
if nargin ~= 2
    error('dace_model requires 2 input arguments')
end

% Check design points
[m1 nx] = size(X);
[m2 ny] = size(Y);
if m1 ~= m2
    error('X and Y must have the same number of rows')
end

theta = 10 * ones(1, nx);
range = minmax(X');

[dmodel, perf] = dacefit(X, Y, @regpoly0, @corrgauss, theta, range(:,1), range(:,2));
end

function [x] = denormalize(surr, x_normal)
% DENORMALIZE() de-normalizes x values from [eps,1]
%

eps = 1.e-4;

nx = length(surr.range);
x = zeros(size(x_normal));

for i = 1:nx
    x(:,i) = min(range{i}) + x_normal(:,i) * (max(range{i}) - min(range{i})) / (1-eps);
end
end

function [flag] = is_valid(surr)
flag = ~isempty(surr.model_data) & sum(isinf(surr.model_data.error)) == 0;
end

function y = mlp_predict(X, mmodel)
% MLPPREDICT - To predict using MLP model
%
% Call
%    y = mlp_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : MLP model obtained using mlp_model
%
% Output:
% y   : Predicted response
%

% Check arguments
if nargin ~= 2
    error('mlp_predict requires 2 input arguments')
end

y = sim(mmodel, X')';
end

function mmodel = mlp_train(X, Y)
% MLPMODEL - To construct MLP model
%
% Call
%    rmodel = mlp_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
%
% Output
% rmodel : MLP Model
%

% Check arguments
if nargin ~= 2
    error('mlp_model requires 2 input arguments')
end

% Check design points
[m1 nx] = size(X);
[m2 ny] = size(Y);
if m1 ~= m2
    error('X and Y must have the same number of rows')
end

ny = size(Y, 2);

%range = minmax(X');
%neuron = [5, 1];
actfunc = {'tansig' 'tansig'};
%mmodel = newff(range, neuron, actfunc, 'trainlm');

mmodel = newff(X', Y', 6, actfunc);
mmodel.trainParam.epochs = 5000;
% mmodel.trainParam.goal = 1.e-2;
mmodel.trainParam.showWindow = 0;

% avoid splitting data into train/validate/test sets
% mmodel.divideFcn = '';
mmodel = train(mmodel, X', Y');
end

function[surr_m,id1m,id2m] = modified_surr(surr,id1,id2,modelnum)
surry = surr.y(:,modelnum);
surr_m = surr;
idreject = union(find(isnan(surry)),find(isinf(surry)),'stable');
id1m = setdiff(id1,idreject,'stable');
id2m = setdiff(id2,idreject,'stable');
require = surr.nx + 1 - numel(id1m);
available = numel(id2m);
if numel(id1m) < surr.nx+1 && available > 0
    leastavail = min(require,available);
    id1m = [id1m;id2m(1:leastavail,:)];
    id2m = id2m(leastavail+1:end,:);
end
end

function [max_nrmse,varargout] = normal_rms_error(surr, y, ypred)

[N, m] = size(y);
nrmse = zeros(1,m);
for i = 1:m
    ydiff = y(:,i) - ypred(:,i);
    mse = (ydiff' * ydiff) / N;
    rmse = sqrt(mse);
    mm = minmax(y');
    delta = mm(2) - mm(1);
    if delta < 1.e-6, delta = 1; end
    nrmse(i) = rmse / delta;
end
max_nrmse = max(nrmse);

% Update based on Kendall Tau
%     [temp,I]=sort(y);
%     [temp,J]=sort(ypred);
%     k_tau=corr(I,J,'type','kendall');
%     max_nrmse=(1-k_tau)/2;

if nargout == 2
    varargout{1} = nrmse;
end
end

function [x_normal] = normalize(surr, x)
% NORMALIZE() normalizes x value between [eps,1]

eps = 1.e-4;

nx = length(surr.range);
x_normal = zeros(size(x));

range = surr.range;
for i = 1:nx
    if range(i,1) == range(i,2)
        x_normal(:,i) = ones(size(x,1),1);
    else
        x_normal(:,i) = eps + (x(:,i) - min(range(i,:))) * (1-eps) / (max(range(i,:)) - min(range(i,:)));
    end
    %         x_normal(:,i) = x(:,i);
end
end

function [y, valid] = predict(surr, x)
y = zeros(size(x,1), surr.ny);
valid = zeros(size(x,1),1);
for i = 1:size(x,1)
    [yp, valid(i)] = predict_cluster(surr, x(i,:));
    if valid(i)
        y(i,:) = yp;
    end
end
end

function [y, valid] = predict_cluster(surr, x)

assert(size(x,1) == 1, 'Predict_cluster() requires only one point');

[valid, c_id] = validate(surr, x);
if valid
    csdata = surr.model_data.csdata{c_id};
    [y, valid] = predict_model(surr, csdata.type, csdata.model, x);
else
    y = [];
end
end

function [y, valid] = predict_model(surr, type, model, x)

N = size(x,1);
m = length(type);

y = zeros(N, m);

valid = 1;
for i = 1:m
    if ~isempty(model{i})
        y(:,i) = predict_model_single(surr, type{i}, model{i}, x);
    else
        valid = 0;
        break
    end
end
end


function [y] = predict_model_single(surr, type, model, x)
x = normalize(surr, x);
ptype = type;
pred_func = strcat(ptype, '_predict');
y = zeros(size(x,1), 1);
for i = 1:size(x,1)
    y(i) = feval(pred_func, x(i,:), model);
end
end

function y = rbf_predict(X, rbmodel)
% RBFPREDICT - To predict using radial basis function model
%
% Call
%    y = rbf_predict(X, rbmodel)
%
% Input
% X      : Data Points
% rmodel : Radial Basis model obtained using rbf_model
%
% Output:
% y   : Predicted response
%

% Check arguments
if nargin ~= 2
    error('rbf_predict requires 2 input arguments')
end

y = sim(rbmodel, X')';
end

function rbmodel = rbf_train(X, Y)
% RBFMODEL - To construct radial basis function network model
%
% Call
%    rbmodel = rbf_train(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
%
% Output
% rbmodel : RBF Model
%

% Check arguments
if nargin ~= 2
    error('rbf_model requires 2 input arguments')
end

% Check design points
[m1 nx] = size(X);
[m2 ny] = size(Y);
if m1 ~= m2
    error('X and Y must have the same number of rows')
end

rbmodel = newrbe(X', Y');
end

function  [f, df] = regpoly0(S)
%REGPOLY0  Zero order polynomial regression function
%
% Call:    f = regpoly0(S)
%          [f, df] = regpoly0(S)
%
% S  : m*n matrix with design sites
% f  : ones(m,1)
% df : Jacobian at the first point (first row in S)

% hbn@imm.dtu.dk
% Last update  April 12, 2002

[m n] = size(S);
f = ones(m,1);
if  nargout > 1
    df = zeros(n,1);
end
end

function  [f, df] = regpoly1(S)
%REGPOLY1  First order polynomial regression function
%
% Call:    f = regpoly1(S)
%          [f, df] = regpoly1(S)
%
% S : m*n matrix with design sites
% f = [1  s]
% df : Jacobian at the first point (first row in S)

% hbn@imm.dtu.dk
% Last update April 12, 2002

[m n] = size(S);
f = [ones(m,1)  S];
if  nargout > 1
    df = [zeros(n,1) eye(n)];
end
end

function  [f, df] = regpoly2(S)
%REGPOLY2  Second order polynomial regression function
% Call:    f = regpoly2(S)
%          [f, df] = regpoly2(S)
%
% S : m*n matrix with design sites
% f =  [1 S S(:,1)*S S(:,2)S(:,2:n) ... S(:,n)^2]
% df : Jacobian at the first point (first row in S)

% hbn@imm.dtu.dk
% Last update September 4, 2002

[m n] = size(S);
nn = (n+1)*(n+2)/2;  % Number of columns in f
% Compute  f
f = [ones(m,1) S zeros(m,nn-n-1)];
j = n+1;   q = n;
for  k = 1 : n
    f(:,j+(1:q)) = repmat(S(:,k),1,q) .* S(:,k:n);
    j = j+q;   q = q-1;
end

if  nargout > 1
    df = [zeros(n,1)  eye(n)  zeros(n,nn-n-1)];
    j = n+1;   q = n;
    for  k = 1 : n
        df(k,j+(1:q)) = [2*S(1,k) S(1,k+1:n)];
        for i = 1 : n-k,  df(k+i,j+1+i) = S(1,k); end
        j = j+q;   q = q-1;
    end
end
end

function y = rsm1_predict(X, rmodel)
% RSMPREDICT - To predict using quadratic response surface model
%
% Call
%    y = rsm_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : Response surface model obtained using rsm_model
%
% Output:
% y   : Predicted response
%

% Check arguments
if nargin ~= 2
    error('rsm_predict requires 2 input arguments')
end

% Check data points
[m nx] = size(X);
if nx ~= rmodel.nx
    error('X must be consistent with rmodel')
end

% Construct F matrix for prediction
F = regpoly1(X);
y = F * rmodel.b;
end

function rmodel = rsm1_train(X, Y)
% RSMMODEL - To construct quadratic response surface model
%
% Call
%    rmodel = rsm_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
%
% Output
% rmodel : RSM Model - a struct with elements
%

% Check arguments
if nargin ~= 2
    error('rsm_model requires 2 input arguments')
end

% Check design points
[m1 nx] = size(X);
[m2 ny] = size(Y);
if m1 ~= m2
    error('X and Y must have the same number of rows')
end

% Construct F matrix for quadratic model
F = regpoly1(X);
nf = size(F, 2);

% Regression model
b = zeros(nf, ny);
for i = 1:ny
    F = regpoly1(X);
    b(:,i) = regress(Y(:,i), F);
end

rmodel = struct('nx', nx, 'ny', ny, 'b', b);
end

function y = rsm2_predict(X, rmodel)
% RSMPREDICT - To predict using quadratic response surface model
%
% Call
%    y = rsm_predict(X, rmodel)
%
% Input
% X      : Data Points
% rmodel : Response surface model obtained using rsm_model
%
% Output:
% y   : Predicted response
%

% Check arguments
if nargin ~= 2
    error('rsm_predict requires 2 input arguments')
end

% Check data points
[m nx] = size(X);
if nx ~= rmodel.nx
    error('X must be consistent with rmodel')
end

% Construct F matrix for prediction
F = regpoly2(X);
y = F * rmodel.b;
end

function rmodel = rsm2_train(X, Y)
% RSMMODEL - To construct quadratic response surface model
%
% Call
%    rmodel = rsm_model(X, Y)
%
% Input
% X  : Data Points X(i,:), i=1,...,m
%
% Output
% rmodel : RSM Model - a struct with elements
%

% Check arguments
if nargin ~= 2
    error('rsm_model requires 2 input arguments')
end

% Check design points
[m1 nx] = size(X);
[m2 ny] = size(Y);
if m1 ~= m2
    error('X and Y must have the same number of rows')
end

% Construct F matrix for quadratic model
F = regpoly2(X);
nf = size(F, 2);

% Regression model
b = zeros(nf, ny);
for i = 1:ny
    F = regpoly2(X);
    b(:,i) = regress(Y(:,i), F);
end

rmodel = struct('nx', nx, 'ny', ny, 'b', b);
end

function [surr] = set_logger(surr, logger)
% SET_LOGGER() Set the logger instance
%
surr.logger = logger;
end

function [surr] = set_range(surr, range)
% SET_RANGE() sets the range of decision variables
nx = size(range,1);
surr.range = range;
surr.nx = nx;
end

function [surr] = Surrogate(param)
% SURROGATE() creates surrogate model
%
%  There are 5 steps to building surrogate models and using them
%
%  1. Create surrogate class with appropriate parameters
%  2. Set the variable ranges (for normalization)
%  3. Add points to the archive
%  4. Train the surrogate model
%  5. Use the surrogate model for prediction
%
%Example: building a surrogate model for y = f(x)
%
%  surr = Surrogate();
%  surr = set_range(surr, Range('range', [0,1]));
%  x = rand(10,1);
%  y = some_func_to_approximate(x);
%  surr = add_points(surr, x, y);
%  surr = trainsurr(surr);
%  y_pred = predict(surr, x);
%
%Parameters:
%
%  surr_type  - type of surrogate model { 'rsm', 'rbf', 'dace', 'mlp' }
%               To add a new type, check surrogate/ subdirectory.
%
%  surr_num_clusters   - number of clusters (0 for adaptive)
%  surr_add_crit       - epsilon neighborhood to add new points to archive
%  surr_train_ratio    - fraction of solutions used for training
%  surr_max_traincount - maximum no. of points used for training
%  surr_mse_threshold  - surrogate validity criteria
%  surr_pred_dist      - fraction of solid diagonal to prevent extrapolation
%

surr.range = [];
surr.x = [];
surr.x_normal = [];
surr.y = [];
surr.nx = 0;
surr.ny = 0;
surr.count = 0;
surr.logger = [];

surr.n_clusters = 1;
surr.type = { 'rbf' };
surr.adaptive_clustering = 0;
surr.seed = 1;
surr.max_traincount = 200;
surr.train_ratio = 0.8;
surr.add_crit = 1.e-3;
surr.pred_dist = 0.05;


% model data
%   model_data.cluster  - data clusters
%	model_data.error(i) - max prediction error for each cluster
%	model_data.csdata{i} - surrogate models for each cluster
% 	  model_data.csdata{i}.type{j} - surrogate model type for jth response
%	  model_data.csdata{i}.model{j} - surrogate model parameters
%	  model_data.csdata{i}.error(j) - prediction error
surr.model_data = [];

% 	% load parameters
% 	param_value = [];
% 	if nargin == 1, param_value = varargin{1}; end
% 	param = Paramset(param_value);
% 	param = add(param, 'seed', Range('irange', [1,2^32-1]));
%
% 	param = add(param, 'surr_num_clusters', Range('irange', [0,100]));
% 	param = add(param, 'surr_type', ...
% 		Range('subset', {'rsm', 'orsm', 'rbf', 'orbf', 'dace', 'mlp', 'omlp'}));
% 	param = add(param, 'surr_max_traincount', Range('irange', [1, 10000]));
% 	param = add(param, 'surr_train_ratio', Range('range', [0.5,1]));
% 	param = add(param, 'surr_add_crit', Range('range', [0,1]));
% 	param = add(param, 'surr_mse_threshold', Range('range', [0,1]));
% 	param = add(param, 'surr_pred_dist', Range('range', [0,1]));
% 	param = check(param);

% Assign parameter values
surr.seed = param.seed;
surr.n_clusters = param.surr_num_clusters;
if surr.n_clusters == 0
    surr.adaptive_clustering = 1;
end
surr.type = param.surr_type;
surr.max_traincount = param.surr_max_traincount;
surr.train_ratio = param.surr_train_ratio;
surr.add_crit = param.surr_add_crit;
surr.tau_err_threshold = param.surr_tau_err_threshold;
surr.pred_dist = param.surr_pred_dist;
surr.choose_surr = param.choose_surr;

% 	surr = class(surr, 'Surrogate');
end

function [surr] = trainsurr(surr, param, prob)
% TRAIN() trains surrogate model(s)

% Save random state
rand_state = rng(param.seed,'twister');

% First step is construct number of clusters
% Second step is build surrogate for each cluster

if ~surr.adaptive_clustering
    surr.model_data = train_cluster(param,surr, surr.n_clusters, prob);
else
    % heuristic on max number of clusters
    max_clusters = round(sqrt(surr.count/5)/4);
    
    best_k = 0;
    best_maxerror = Inf;
    best_model_data = [];
    for i = 1:max_clusters
        surr.n_clusters = i;
        surr.model_data = train_cluster(param,surr, i, prob);
        maxerror = max(surr.model_data.error);
        writesurr(param,surr, '\tTrying %d clusters: Valid=%d, Error=%g\n', i, ...
            sum(~isinf(surr.model_data.error)), maxerror);
        if i == 1 || maxerror < best_maxerror
            best_k = i;
            best_maxerror = maxerror;
            best_model_data = surr.model_data;
        end
    end
    
    surr.model_data = best_model_data;
    surr.n_clusters = best_k;
end

id = find(~isinf(surr.model_data.error));
if isempty(id)
    n_valid = 0;
    maxerror = inf;
else
    n_valid = length(id);
    maxerror = max(surr.model_data.error(id));
end

writesurr(param,surr, '\tSurrogate: cluster=%d/%d, error=%g\n', ...
    n_valid, surr.n_clusters, maxerror);

% Restore random state
rng(rand_state);
end

function [model_data] = train_cluster(param,surr, k, prob)

model_data.cluster = Cluster(surr.x, surr.seed, 'k_means', k);
model_data.csdata = cell(k,1);
model_data.error = zeros(k,1);
model_data.allerror = NaN*ones(k,surr.ny);

% for each cluster
for i = 1:k
    ids = model_data.cluster.id{i};
    
    % identify the points used to train
    n_total = length(ids);
    traincount = round(n_total * surr.train_ratio);
    if traincount > surr.max_traincount
        traincount = surr.max_traincount;
        n_max = min(n_total, round(traincount / surr.train_ratio));
        ids = ids(n_total-n_max+1:end);
    else
        n_max = traincount;
    end
    writesurr(param,surr, '\tCluster %d: %d/%d\n', i, n_max, n_total);
    
    cdata = Cluster(surr.x(ids,:), surr.seed, 'k_medoids', traincount);
    t_ids = ids(cdata.centroid);
    v_ids = setdiff(ids, t_ids);
    
    % for each response
    csdata =[];
    csdata.type = cell(1,surr.ny);
    csdata.model = cell(1,surr.ny);
    csdata.error = Inf * ones(1,surr.ny);
    for j = 1:surr.ny
        [csdata.type{j}, csdata.model{j}, csdata.error(j)] = ...
            train_surr_model(param,surr, t_ids, v_ids, j, k);
        if isinf(csdata.error(j))
            break
        end
    end
    model_data.type{i} = csdata.type;
    model_data.csdata{i} = csdata;
    model_data.error(i) = max(csdata.error);
    model_data.allerror(i,:) = csdata.error;
end
end

function [best_m_type, best_m_data, best_m_nrmse] = train_surr_model(param,surr, t_ids, v_ids, r_id, k)

n = length(surr.type);
best_m_type = [];
best_m_data = [];
best_m_nrmse = [];
found = 0;
writesurr(param,surr, '\t    y(%d): ', r_id);
for i = 1:n
    m_type = surr.type{i};
    [m_data, m_nrmse] = train_model_single(surr, t_ids, v_ids, m_type, r_id);
    writesurr(param,surr, '%s (%g), ', m_type, m_nrmse);
    if surr.choose_surr == 1
        if m_nrmse <= surr.tau_err_threshold
            best_m_type=m_type;
            best_m_data=m_data;
            best_m_nrmse=m_nrmse;
            found = 1;
            break
        else
            if(i==1)
                best_m_nrmse=m_nrmse;
            end
            if m_nrmse <= best_m_nrmse
                best_m_type=m_type;
                best_m_data=m_data;
                best_m_nrmse=m_nrmse;
                found=1;
            end
        end
    else
        if(i==1)
            best_m_nrmse=m_nrmse;
        end
        if m_nrmse <= best_m_nrmse
            best_m_type=m_type;
            best_m_data=m_data;
            best_m_nrmse=m_nrmse;
            found=1;
        end
    end
end
writesurr(param,surr, '\n');

if ~found
    best_m_nrmse = inf;
end
end

function [model, nrmse] = train_model_single(surr, t_ids, v_ids, type, r_id)
[surry,t_ids_m,v_ids_m] = modified_surr(surr,t_ids,v_ids,r_id);
x = surry.x_normal;
ttype = type;
train_func = strcat(ttype, '_train');
warning('off', 'all');
model = feval(train_func, x(t_ids_m,:), surry.y(t_ids_m,r_id));
warning('on', 'all');

% calculate normalized RMSE
if ~isempty(v_ids_m)
    [y, valid] = predict_model(surry, {type}, {model}, surry.x(v_ids_m,:));
    nrmse = normal_rms_error(surry, surry.y(v_ids_m,r_id), y);
else
    nrmse = 0;
end
end

% Check if the surrogate model is valid for given x
function [valid, c_id] = validate(surr, x)

valid = 0;
c_id = 0;

if surr.n_clusters == 1
    if ~isinf(surr.model_data.error(1))
        valid = 1;
        c_id = 1;
    end
else
    x_normal = normalize(surr, x);
    dist = zeros(1, surr.n_clusters);
    for i = 1:surr.n_clusters
        c = surr.model_data.cluster.centroid(i,:);
        c_normal = normalize(surr, c);
        tmp = x_normal - c_normal;
        dist(i) = sqrt(tmp * tmp');
    end
    [mindist, I] = min(dist);
    if mindist < surr.pred_dist * sqrt(surr.nx)
        if ~isinf(surr.model_data.error(I))
            c_id = I;
            valid = 1;
        end
    end
    
    % 	x_normal = normalize(surr, x);
    % 	if find_closest(surr, x_normal) < surr.pred_dist * sqrt(surr.nx)
    % 		max_nrmse = zeros(1, surr.n_clusters);
    % 		id = find_neighbors(surr, x, 10);
    % 		xv = surr.x(id,:);
    % 		yv = surr.y(id,:);
    % 		for j = 1:surr.n_clusters
    % 			type = surr.model_data.csdata{j}.type;
    % 			model = surr.model_data.csdata{j}.model;
    % 			[yv_pred, valid] = predict_model(surr, type, model, xv);
    % 			if valid
    % 				max_nrmse(j) = normal_rms_error(surr, yv, yv_pred);
    % 			else
    % 				max_nrmse(j) = inf;
    % 			end
    % 		end
    % 		[tmp, c_id] = min(max_nrmse);
    % 		if ~isinf(tmp)
    % 			valid = 1;
    % 		end
    % 	end
    
end
end

function [min_dist] = find_closest(surr, x_normal)
tmp = surr.x_normal - repmat(x_normal, surr.count, 1);
dist = sum(tmp .* tmp, 2);
[tmp] = min(dist);
min_dist = tmp(1);
end


function [id] = find_neighbors(surr, x, n)
tmp = surr.x - repmat(x, surr.count, 1);
dist = sum(tmp .* tmp, 2);
[tmp, I] = sort(dist);
id = I(1:n);
end

%% nd_sort() - Find non-dominated points from a given set
% dir = 1, minimize all values
% dir = 2, maximimze all values
function [ndset, idx] = nd_rank1(set1, dir)
if nargin == 1
    dir = 1;
end
[N,M] = size(set1);
switch(dir)
    case 1
        dom = nd_sort_min(set1, M, N);
    case 2
        dom = nd_sort_max(set1, M, N);
    otherwise
        error('wrong value of dir');
end

idx = [];
for i = 1:N
    if dom(i) == 0, idx = [idx i]; end
end
ndset = set1(idx(:),:);
end

%% N objective ND sort
function [ranks] = sort_cluster(f, cdata)

k = length(cdata.id);
best = zeros(1, k);

% Sort each cluster
max_size = 0;
for i = 1:k
    id = cdata.id{i};
    [tmp, I] = sort(f(id));
    c_ranks{i} = id(I);
    best(i) = id(I(1));
    if length(id) > max_size
        max_size = length(id);
    end
end

% Sort the best from each cluster
[tmp, c_order] = sort(f(best));

ranks = [];
for i = 1:max_size
    for j = 1:k
        c = c_order(j);
        if length(c_ranks{c}) >= i
            ranks = [ranks; c_ranks{c}(i)];
        end
    end
end

end

%% N objective ND sort
function [ranks, nd_rank, crowd] = sort_obj(f, id)

[N, M] = size(f);

nd_rank = zeros(N,1);
crowd = zeros(N,1);

if M == 1
    [tmp, I] = sort(f(id));
    ranks = id(I);
else
    ranks = [];
    F = nd_sort(f, id);
    for front = 1 : length(F)
        nd_rank(F(front).f(:)) = front;
        [r1, d1] = sort_crowding(f, F(front).f);
        crowd(F(front).f(:)) = d1;
        ranks = [ranks; r1];
    end
end
end


