%MC_DEMO_SOLVE_ADMM     Solve matrix completion on graphs with ADMM
%
%
%
% see also: MC_solve_ADMM, MC_demo_grid_search, split_observed,
%           sample_sparse, sample_sparse_t, sample_sparse_AtA
%
%code author: Vassilis Kalofolias
%date: Nov 2014

load MC_community_example Gu Gm Xn

Gc = Gu;        % columns graph
Gr = Gm;        % rows graph

%% best settings found for 20% observations (with grid search): 
%% [n=3,   no graphs]:          error = 0.97
%% [n=.01, r=.003, c=.003]:     error = 0.88

% Keep 20% for training, the rest for validation
[y_train, mask_train, y_val, mask_val, y_test, mask_test] = split_observed(Xn, [.2, .8, 0]);

params.size_X = size(Xn);



%if ~isfield(params, 'zero_mean'), params.zero_mean = 1; end     % this should be true for nuclear norm in general!!
%if ~isfield(params, 'maxit'), params.maxit = 50; end         % how many iterations?
%if ~isfield(params, 'verbose'), params.verbose = 0; end
%if ~isfield(params, 'single'), params.single = isa(y_train, 'single'); end


%% Normalize data to zero mean and keep the linear transformation details
y_lims_init = [min(y_train), max(y_train)];

mean_train = mean(y_train);

y_train = y_train - mean_train;
y_val = y_val - mean_train;
%y_test = y_test - mean_train;

y_lims_scaled = [min(y_train), max(y_train)];

%% PREPARE PROBLEM PARAMS
% GRAPHS: (normalized)
prob_params.Lc = (single(full(Gc.L))/Gc.lmax);
prob_params.Lr = (single(full(Gr.L))/Gr.lmax);

%prob_params.Gc_lmax = 1;
%prob_params.Gr_lmax = 1;

% DATASETS and masks:
prob_params.size_X = params.size_X;
prob_params.mask_val = mask_val;
prob_params.mask_test = mask_test;
prob_params.A_op = @(x) sample_sparse(x, mask_train);
prob_params.At_op = @(x) sample_sparse_t(x, mask_train);
prob_params.AtA_op = @(x) sample_sparse_AtA(x, mask_train);


%% SOLVER PARAMETERS 
solver_params.maxit = 100;
solver_params.verbose = 3;

solver_params.tol_abs = 2e-6;
solver_params.tol_rel = 1e-5;

% need the scaling used for preprocessing to calculate error correctly
solver_params.y_lims_init = y_lims_init;
solver_params.y_lims_scaled = y_lims_scaled;

% MOST IMPORTANT: use verbose = 1 to set rho accordingly (depends on tolerances)
%solver_params.rho_ADMM = .005000;
%solver_params.rho_ADMM = .2 * geomean([max(1e-3,prob_params.gamma_n), geomean([max(1e-3,norm(y_train)), max(1e-3,prob_params.gamma_r), max(1e-3,prob_params.gamma_c)])]);

% for small matrices use false!
solver_params.svds = false;



%% Solve the problem using graphs
prob_params.gamma_n = .01;
prob_params.gamma_r = .003;
prob_params.gamma_c = .003;
solver_params.rho_ADMM = .009;

[X_MC_graphs, stat_MC_graphs] = MC_solve_ADMM(y_train, y_val, y_test, prob_params, solver_params);


%% Solve without graphs, just low rank information
prob_params.gamma_n = 3;
prob_params.gamma_r = 0;
prob_params.gamma_c = 0;
solver_params.rho_ADMM = .15;

% Now ADMM is equivalent to forward backward algorithm!
[X_MC_low_rank, stat_MC_low_rank] = MC_solve_ADMM(y_train, y_val, y_test, prob_params, solver_params);
rmse_0 = sqrt(mean((X_MC_low_rank(mask_val) - y_val).^2)); %just for test

%% 
figure;
plot(stat_MC_low_rank.rmse_val);
hold all
plot(stat_MC_graphs.rmse_val);
legend('low rank', 'low rank + graphs')
title('Validation error of different models')
xlabel('iteration')
ylabel('RMSE')


%% TO PLOT RECOVERED AND INITIAL MATRICES:
figure; imagesc(Xn); title('Ground truth (close to low rank with community structure')
figure; imagesc(reshape(prob_params.At_op(y_train), params.size_X)); title('20% observations used for recovery')
figure; imagesc(X_MC_graphs); title('Recovered matrix from 20% observations, using graph and low rank information')

%% To reshape in matrix form use:
% Y_train =   reshape(prob_params.At_op(y_train), params.size_X);
% Y_val =     reshape(sample_sparse_t(y_val, mask_val), params.size_X);
% Y_test =    reshape(sample_sparse_t(y_test, mask_test), params.size_X);




