% generate random data to test
% generate the graphs with gsp toolbox
% gsp_start;
% init_unlocbox;
clear
%% Load Dataset
% Import covariate data and create kNN graphs
% addpath('C:\Users\eee\Desktop\BMCG\dataset\MovieLens\ml-100k');
[Gu,Gv,m,n] = createGraphsFromMovieLensCovariateData(); % k = 10;
% Create Laplacian matrices
Lu = laplacian(graph(Gu));
Lv = laplacian(graph(Gv));

% Import user rating data
U1Base = tdfread('u1.base','|'); % Take one CV split for demo
U1Test = tdfread('u1.test','|');
% {i,j,r} tuples as sparse matrix representation
R_train=U1Base.user_id_item_id_rating_timestamp(:,1:3);
R_val=U1Test.user_id_item_id_rating_timestamp(:,1:3);

% Normalise data (optional)
% R_train(:,3)=R_train(:,3)./5;
% R_val(:,3)=R_val(:,3)./5;

R_train=sparse(R_train(:,1),R_train(:,2),R_train(:,3),m,n);
R_val=sparse(R_val(:,1),R_val(:,2),R_val(:,3),m,n);

M = R_train;
Val = R_val;
% M = full(R_train);
% Val = full(R_val);

O = ones(size(M)) & M;
mask_train = O(:);
mask_val = ones(size(Val)) & Val;
mask_test = [];
M_train = M(mask_train);
M_val = Val(mask_val);
M_test = [];
%% Generate graphs (see GSPBOX for details)
% graph_type = 3; %1:factor graphs;2:column/row graphs;3:obseved matrix graphs
% param_graph.k = 10;
% if graph_type == 1
%     Gu = gsp_nn_graph(U0, param_graph);
%     Gv = gsp_nn_graph(V0, param_graph);
%     Lu = Gu.L;
%     Lv = Gv.L;
% % Lu = Gu.L/max(Lu(:));
% % Lv = Gv.L/max(Lv(:));
% elseif graph_type == 2
%     Gu = gsp_nn_graph(Xn, param_graph);
%     Gv = gsp_nn_graph(Xn', param_graph);
%     Lu = Gu.L;
%     Lv = Gv.L;
% elseif graph_type == 3
%     Gu = gsp_nn_graph(M, param_graph);
%     Gv = gsp_nn_graph(M', param_graph);
%     Lu = Gu.L;
%     Lv = Gv.L;
% end
%% Missing data
% ObsRatio = 0.2;
% [~, mask_train, M_val, mask_val, M_test, mask_test] = split_observed(Xn, [ObsRatio, 1-ObsRatio, 0]);
% [m,n] = size(Xn);
% O = reshape(mask_train,m,n);
% VAL = reshape(mask_val,m,n);
% Val = VAL.*Xn;
% R_val = sparse(VAL.*Xn);
%% Add noise; Generate noisy observation M
% noiseless = 0;
% if noiseless == 0
%     SNR = 10;                     % Noise levels
%     sigma2 = var(Xn(:))*(1/(10^(SNR/10)));
%     GN = sqrt(sigma2)*randn(m,n);
%     M = Xn + GN;                 % fully observed noisy data
%     R_train = sparse(O.*M);                   % same as M_train, aka partially observed data
%     M_train = R_train(mask_train);
%     fprintf('\n----------Demo: Matrix Completion on Graphs----------\n')
%     fprintf('Observation ratio = %g, SNR = %g, True Rank= %d \n', ObsRatio, SNR, R);
%     fprintf(resultsLogFID, '\nNew experiment on graphmatrix data: Observation ratio = %g, SNR = %g, True Rank= %d \n %s .... \n', ObsRatio, SNR, R,datetime('now'));
% else
%     R_train = sparse(O.*Xn); % noiseless case
%     M_train = R_train(mask_train);
%     fprintf('\n----------Demo: Matrix Completion on Graphs----------\n')
%     fprintf('Observation ratio = %g, Noiseless, True Rank= %d \n', ObsRatio, R);
% end
%% Proposed
ts = tic;
[Result, stat] = BMCG_update(M, 'obs', O, 'initRank',10,'maxiters',100, 'Gu', Lu, 'Gv', Lv, 'val', Val, 'V', mask_val);
t_total = toc(ts);
X_est = Result.X;
err = X_est(mask_val) - M_val;
rmse = sqrt(mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(M_val.^2));
fprintf('\n----------With BMCG update Pruning----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, Time = %g\n', ...
    full(rrse), full(rmse), Result.EstRank, t_total);
%% Comparison
D = 10;
%% Simple benchmark scores
[global_mean_val_error,col_mean_val_error,row_mean_val_error,R_val_i,R_val_j,R_val_v]=averagingRMatrixPredictionScores(resultsLogFID,R_train,R_val,dimN,dimM);
%% GRALS - full graph
GRALS_Comp_Time=tic;

CG_iter=2;
max_algo_iters=200;
gamma_U=0.0001;
gamma_V=0.0001;
gamma_L=0.4;
Gral_config = ['-k ',num2str(dimD),' -e 0.001 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];

% regularise Laplacian matrices (Section 2, Laplacian Matrix of GPMF paper, and Equation 5 of GRALS paper)
L_U_reg = gamma_L*L_U + gamma_U*speye(dimN);
L_V_reg = gamma_L*L_V + gamma_V*speye(dimM);
    
[U_GRALS,V_GRALS,RMSE_GRALS,WallTime_GRALS] = glr_mf_train(R_train,R_val,L_U_reg,L_V_reg,Gral_config);
U_GRALS = U_GRALS';
V_GRALS = V_GRALS';
GRALSCompT=toc(GRALS_Comp_Time);

rmse_GRALS(loopidx,Rindex) = sqrt( (length(R_val_v)^-1)*sum((sum(U_GRALS(R_val_i,:) .* V_GRALS(R_val_j,:),2) - R_val_v).^2) );
fprintf(resultsLogFID, 'GRALS: %f \n',rmse_GRALS(loopidx,Rindex));
% disp(['GRALS RMSE: ',num2str(GRALS_val_error)]);
% fprintf(resultsLogFID, '\n===\nGRALS, reg. Laplacian (gamma_L=%f,gamma_U=%f,gamma_V=%f) \nval: %f (%.2f secs.)\n',...
%     gamma_L,gamma_U,gamma_V,GRALS_val_error,GRALSCompT);
% fprintf(resultsLogFID, '\nSettings: D=%i, CG iter=%i\n===\n\n',dimD,CG_iter);
% 
% 
% fprintf(resultsLogFID, '\nSettings: D=%i\n===\n\n',dimD);

%% GPMF - Use stored PMF values for initialisation, run M-step, then E-step once
GPMF_Comp_Time=tic;

% Initialise U,V with no graph (PMF)
GPMF_Comp_Time_init=tic;

CG_iter=3;
max_algo_iters=35;
Gral_config = ['-k ',num2str(dimD),' -e 0.001 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];
Lap_diag_gamma=1.2;
gamma_U_diag=Lap_diag_gamma*speye(dimN);
gamma_V_diag=Lap_diag_gamma*speye(dimM);

[U_GPMF_INIT,V_GPMF_INIT,RMSE_GPMF_INIT,WallTime_GPMF_INIT] = glr_mf_train(R_train,R_val,gamma_U_diag,gamma_V_diag,Gral_config);
U_GPMF_INIT = U_GPMF_INIT';
V_GPMF_INIT = V_GPMF_INIT';

GPMFCompTinit=toc(GPMF_Comp_Time_init);

%% M-step: Approximate posterior covariance with diagonalized row-wise covariance
GPMF_Comp_Time_MApproxPost_Full=tic;

sig2=0.05;
alph=sig2^-1;

GPMF_Comp_Time_MApproxPost_U=tic;
SPostHat_U_ds = EstimatePosteriorCovarianceForMStep(V_GPMF_INIT,sparse(dimN,dimN),dimN,R_val_j,R_val_i,dimD,alph,L_U);
GPMFCompTMApproxPost_U=toc(GPMF_Comp_Time_MApproxPost_U);

GPMF_Comp_Time_MApproxPost_V=tic;
SPostHat_V_ds = EstimatePosteriorCovarianceForMStep(U_GPMF_INIT,sparse(dimM,dimM),dimM,R_val_i,R_val_j,dimD,alph,L_V);
GPMFCompTMApproxPost_V=toc(GPMF_Comp_Time_MApproxPost_V);

GPMFCompTMApproxPost_Full=toc(GPMF_Comp_Time_MApproxPost_Full);

disp('Approximate posterior covariance estimates done.')

%% M-step: Compute outer produce of MAP estimates of latent faetures 

%(only compute outer products matching non-zeros in the adjacency matrix,
% as significantly less to compute).

GPMF_Comp_Time_MOuterProductMAP_Full=tic;

U_GPMF_INIT_c = U_GPMF_INIT' - mean(U_GPMF_INIT,2)';  % zero-mean the data
V_GPMF_INIT_c = V_GPMF_INIT' - mean(V_GPMF_INIT,2)';

GPMF_Comp_Time_MOuterProductMAP_U=tic;
S_sparse_U = SparseCovariance(U_GPMF_INIT_c,G_U);  % Embarrassingly parallel potential for extremely large dimension
GPMFCompTMOuterProductMAP_U=toc(GPMF_Comp_Time_MOuterProductMAP_U);

GPMF_Comp_Time_MOuterProductMAP_V=tic;
S_sparse_V = SparseCovariance(V_GPMF_INIT_c,G_V);
GPMFCompTMOuterProductMAP_V=toc(GPMF_Comp_Time_MOuterProductMAP_V);

% M-step expected sample covariance
% GPMF paper - Equation 10 
Expected_sample_cov_U = SPostHat_U_ds + S_sparse_U;
Expected_sample_cov_V = SPostHat_V_ds + S_sparse_V;

%clear SPostHat_U_ds S_sparse_U SPostHat_V_ds S_sparse_V

%% M-step thresholding of expected sample covariance to approximate GLASSO
% solution sparsity structure

GPMF_Comp_Time_Threshold_Full=tic;

tau_U=0.0; % Graph sparsity inducing strength - increase for even sparser solution
tau_V=0.0; 

% Sparse Inverse cov est. 
% ... for U
GPMF_Comp_Time_Threshold_U=tic;

Cov_thr = CovThresholdWithSparsityConstraints(Expected_sample_cov_U,tau_U,U_GPMF_INIT_c);
L_new_U = LaplacianFromMatrixStruct(Cov_thr);
if isempty(L_new_U)
    L_new_U = inv(Cov_thr);
end
Sigma_new_U = Cov_thr;
G_U_thresh = GraphFromMatrixStruct(Cov_thr);

GPMFCompTThreshold_U=toc(GPMF_Comp_Time_Threshold_U);

% ... for V
GPMF_Comp_Time_Threshold_V=tic;

Cov_thr = CovThresholdWithSparsityConstraints(Expected_sample_cov_V,tau_V,V_GPMF_INIT_c);
L_new_V = LaplacianFromMatrixStruct(Cov_thr);
if isempty(L_new_V)
    L_new_V = inv(Cov_thr);
end
Sigma_hat_V = Cov_thr;
G_V_thresh = GraphFromMatrixStruct(Cov_thr);

GPMFCompTThreshold_V=toc(GPMF_Comp_Time_Threshold_V);

GPMFCompTThreshold_Full=toc(GPMF_Comp_Time_Threshold_Full);

% Graph properties to analyse the sparse estimation reduction
L_U_sparsity = nnz(L_U)/(dimN^2);
L_U_new_sparsity = nnz(L_new_U)/(size(L_new_U,1)^2);
L_V_sparsity = nnz(L_V)/(size(L_V,1)^2);
L_V_new_sparsity = (nnz(L_new_V) + dimN)/(size(L_new_V,1)^2);

%% E-step: MAP estimate of U,V with updated graph
GPMF_Comp_Time_E_step=tic;

%CG_iter=2;
CG_iter=2;
max_algo_iters=200;
gamma_U = 0.0001;
gamma_V = 0.0001;
gamma_L=0.4;

Gral_config = ['-k ',num2str(dimD),' -e 0.01 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];

L_U_reg = gamma_L*L_new_U + gamma_U*speye(dimN);
L_V_reg = gamma_L*L_new_V + gamma_V*speye(dimM);

[U_grals_GPMF_graph,V_grals_GPMF_graph,RMSE_GPMF,WallTime_GPMF] = glr_mf_train(R_train,R_val,L_U_reg,L_V_reg,Gral_config);
U_grals_GPMF_graph = U_grals_GPMF_graph';
V_grals_GPMF_graph = V_grals_GPMF_graph';

GPMFCompTEstep=toc(GPMF_Comp_Time_E_step);
GPMFCompT=toc(GPMF_Comp_Time);

rmse_GPMF(loopidx,Rindex) = sqrt( (length(R_val_v)^-1)*sum((sum(U_grals_GPMF_graph(R_val_i,:) .* V_grals_GPMF_graph(R_val_j,:),2) - R_val_v).^2) );
fprintf(resultsLogFID, 'GPMF: %f \n',rmse_GPMF(loopidx,Rindex));

%% BPMF
ts_bpmf = tic;
bpmf_pred = bpmf(M,10);
t_total_bpmf = toc(ts_bpmf);
err_bpmf = bpmf_pred(mask_val) - M_val;
rmse_bpmf = sqrt(mean(err_bpmf.^2));
rrse_bpmf = sqrt(sum(err_bpmf.^2)/sum(M_val.^2));
fprintf('\n----------With BPMF----------\n')
fprintf('RMSE = %g, Time = %g\n', full(rmse_bpmf), t_total_bpmf);
%% ADMM
params.size_X = size(M);

y_lims_init = [min(M_train), max(M_train)];
mean_train = mean(M_train);
y_train = M_train - mean_train;
y_val = M_val - mean_train;
y_lims_scaled = [min(M_train), max(M_train)];

prob_params.Lc = Lv;
prob_params.Lr = Lu;

prob_params.size_X = params.size_X;
prob_params.mask_val = mask_val;
prob_params.mask_test = mask_test;
prob_params.A_op = @(x) sample_sparse(x, mask_train);
prob_params.At_op = @(x) sample_sparse_t(x, mask_train);
prob_params.AtA_op = @(x) sample_sparse_AtA(x, mask_train);

solver_params.maxit = 100;
solver_params.verbose = 3;
solver_params.tol_abs = 2e-6;
solver_params.tol_rel = 1e-5;
solver_params.y_lims_init = y_lims_init;
solver_params.y_lims_scaled = y_lims_scaled;
solver_params.svds = false;

prob_params.gamma_n = .01;
prob_params.gamma_r = .003;
prob_params.gamma_c = .003;
solver_params.rho_ADMM = .009;
% with graphs
ts3 = tic;
[X_MC_graphs, stat_MC_graphs] = MC_solve_ADMM(y_train, y_val, M_test, prob_params, solver_params);
X_MC_graphs = X_MC_graphs + mean_train;
t_total3 = toc(ts3);
err3 = X_MC_graphs(mask_val) - M_val;
rmse3 = sqrt(mean(err3.^2));
rrse3 = sqrt(sum(err3.^2)/sum(M_val.^2));
fprintf('\n----------MC on Graphs with Graph info----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, Time = %g\n', ...
    rrse3, rmse3, stat_MC_graphs.rankX(stat_MC_graphs.last_iter), t_total3);
% without graphs
prob_params.gamma_n = 3;
prob_params.gamma_r = 0;
prob_params.gamma_c = 0;
solver_params.rho_ADMM = .15;%.15
ts4 = tic;
[X_MC_low_rank, stat_MC_low_rank] = MC_solve_ADMM(y_train, y_val, M_test, prob_params, solver_params);
X_MC_low_rank = X_MC_low_rank + mean_train;
t_total4 = toc(ts4);
err4 = X_MC_low_rank(mask_val) - M_val;
rmse4 = sqrt(mean(err4.^2));
rrse4 = sqrt(sum(err4.^2)/sum(M_val.^2));

fprintf('\n----------MC on Graphs without Graph info----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, Time = %g\n', ...
    rrse4, rmse4, stat_MC_low_rank.rankX(stat_MC_low_rank.last_iter), t_total4);


%% L1MC
Max_iter_L1MC = 1000;
ts7 = tic;
X_L1MC = L1MC_Ref(M, O ,min(m,n),10,Max_iter_L1MC);
t_total7 = toc(ts7);
err7 =X_L1MC.X(mask_val) - M_val;
rmse7 = sqrt(mean(err7.^2));
rrse7 = sqrt(sum(err7.^2)/sum(M_val.^2));
fprintf('\n----------L1MC----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse7, rmse7, X_L1MC.r,t_total7);
%% LMaFit
opts = [];
TT = (1 : m*n)';
Known = TT(mask_train);
ts8 = tic;
[X_LMaFit,Y_LMaFit,Out] = lmafit_mc_adp(m,n,D,Known,M_train,opts);
X_est_LMaFit = X_LMaFit*Y_LMaFit;
t_total8 = toc(ts8);
err8 =X_est_LMaFit(mask_val) - M_val;
rmse8 = sqrt(mean(err8.^2));
rrse8 = sqrt(sum(err8.^2)/sum(M_val.^2));
fprintf('\n----------LMaFit----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse8, rmse8, Out.rank,t_total8);
%% ALM-MC
ts9 = tic;
[A,iter,svp] = inexact_alm_mc(M.*O, -1, 1000);
X_ALMMC = A.U*A.V';
t_total9 = toc(ts9);
err9 =X_ALMMC(mask_val) - M_val;
rmse9 = sqrt(mean(err9.^2));
rrse9 = sqrt(sum(err9.^2)/sum(M_val.^2));
fprintf('\n----------ALM-MC----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse9, rmse9, svp,t_total9);
%% BMC-GP-GAMP
Max_it = 100;
W1_inv = eye(min(m,n))*(1e-10);
%W5 = prob_params.Lr+(1e-4)*eye(m);
W5 = prob_params.Lr;
W5_inv = inv(W5);
ts10 = tic;
Result_BMCGPGAMP1 = BMC_GAMP_new(M,O,Max_it,W1_inv);
t_total10 = toc(ts10);
err10 =Result_BMCGPGAMP1.X(mask_val) - M_val;
rmse10 = sqrt(mean(err10.^2));
rrse10 = sqrt(sum(err10.^2)/sum(M_val.^2));
fprintf('\n----------BMC-GP-GAMP without Graphs----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse10, rmse10, rank(Result_BMCGPGAMP1.X),t_total10);

ts11 = tic;
Result_BMCGPGAMP2 = BMC_GAMP_new(M,O,Max_it,W5_inv);
t_total11 = toc(ts11);
err11 =Result_BMCGPGAMP2.X(mask_val) - M_val;
rmse11 = sqrt(mean(err11.^2));
rrse11 = sqrt(sum(err11.^2)/sum(M_val.^2));
fprintf('\n----------BMC-GP-GAMP with Graphs----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse11, rmse11, rank(Result_BMCGPGAMP2.X),t_total11);

%% KPMF
gamma_r = [0.001:0.001:0.01]';
for gamma_idx = 1:length(gamma_r)
method = 'sgd'; % 'gd' or 'sgd'
sigma_r = 0.4;  % Variance of entries
D = 10;           % latent rank 
eta = 0.001;    % Learning rate
gamma=gamma_r(gamma_idx);      % Parameter for graph kernel
% K_v = 0.2 * eye(n);
Graph_V = Gv;
K_v = graphKernel(Graph_V, gamma); 
K_v_inv = inv(K_v);
Graph_U = Gu;
K_u = graphKernel(Graph_U, gamma); 
K_u_inv = pinv(K_u);
ts6 = tic;
[U_KPMF, V_KPMF, vRMSE, time] = kpmf_sgd(M, O, D, K_u_inv, K_v_inv, sigma_r, eta, Val, mask_val);
t_total6 = toc(ts6);
X_est_KPMF = U_KPMF*V_KPMF';
err6 = X_est_KPMF(mask_val) - M_val;
rmse6(gamma_idx) = sqrt(mean(err6.^2));
rrse6 = sqrt(sum(err6.^2)/sum(M_val.^2));
fprintf('\n----------KPMF----------\n')
fprintf('RRSE = %g, RMSE = %g,Time = %g\n', ...
    rrse6, rmse6(gamma_idx),t_total6);
end
%% VBMC
options.verbose = 1;
options.MAXITER = 100; 
options.DIMRED = 1; 
options.UPDATE_BETA = 1;
options.initial_rank = min(m,n); % or we can set a value. 
options.verbose = 1;

ts5 = tic;
[X_hat, A_hat, B_hat] = VBMC(O, M, options);
t_total5 = toc(ts5);


% mse_PMF = sum(sum(err_PMF.^2))/(m*n);
% PSNR_PMF = 20*log10(255/sqrt(mse_PMF)); 
err5 = X_hat(mask_val) - M_val;
rmse5 = sqrt(mean(err5.^2));
rrse5 = sqrt(sum(err5.^2)/sum(M_val.^2));
fprintf('\n----------Variational Bayesian Low Rank Matrix Completion----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, Time = %g\n', ...
    rrse5, rmse5, rank(X_hat), t_total5);

%% Performance Reports
fprintf('\n----------With SVD Pruning----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, \nEstimated SNR = %g, Time = %g\n', ...
    rrse1, rmse1,Result1.EstRank, Result1.SNR, t_total1);
fprintf('\n----------With Maxchange Pruning----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, \nEstimated SNR = %g, Time = %g\n', ...
    rrse2, rmse2,Result2.EstRank, Result2.SNR, t_total2);
fprintf('\n----------With simple SVD Pruning----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, \nEstimated SNR = %g, Time = %g\n', ...
    rrse21, rmse21,Result21.EstRank, Result21.SNR, t_total21);
fprintf('\n----------MC on Graphs with Graph info----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, Time = %g\n', ...
    rrse3, rmse3, stat_MC_graphs.rankX(stat_MC_graphs.last_iter), t_total3);
fprintf('\n----------L1MC----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse7, rmse7, X_L1MC.r,t_total7);
fprintf('\n----------LMaFit----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse8, rmse8, Out.rank,t_total8);
fprintf('\n----------ALM-MC----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse9, rmse9, svp,t_total9);
fprintf('\n----------BMC-GP-GAMP with Graphs----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse11, rmse11, rank(Result_BMCGPGAMP2.X),t_total11);
fprintf('\n----------KPMF----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d,Time = %g\n', ...
    rrse6, rmse6, rank(X_est_KPMF),t_total6);
fprintf('\n----------Variational Bayesian Low Rank Matrix Completion----------\n')
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, Time = %g\n', ...
    rrse5, rmse5, rank(X_hat), t_total5);