function Multirun
startup;
params;
path = pwd;
path11=strcat(path,filesep);
dir=strcat(path11,'Results');
mkdir(dir);
count=1;
pb = param.config;
for i=1:length(pb)
    dir1=strcat(dir,filesep,pb{i});
    mkdir(dir1);
    cd(dir1);
    param.crossover_prob = param.crossover_pr_prob(i);
    param.mutation_prob = param.mutation_pr_prob(i);
    param.mutation_poly_eta = param.mutation_poly_prob(i);
    param.crossover_sbx_eta = param.crossover_sbx_prob(i);
    param.generations=param.gen_prob(i);
    param.problem_name = pb{i};
    param.pop_size = param.popsize_prob(i);
    param.pres_func_eval = param.pres_func_evalall(i);
    for j=1:param.multirun(i)
        dir2=strcat(dir1,filesep,'run-',num2str(j));
        mkdir(dir2);
        cd(dir2);
        param.seed=param.seed_prob(i)+j;
        % Update the definition file
        save('Params.mat', 'param');
        path1{count} = dir2;
        count = count+1;
        cd ..;
    end
    cd ..;
    cd ..;
end
cd(path);
parfor i=1:length(path1)
    cd(path1{i});
    disp(strcat('Running -> ',path1{i}));
    param=load('Params.mat');
    tic; 
    SAMO(param.param);
    toc;
    cd(dir);
end
cd(path);
return

