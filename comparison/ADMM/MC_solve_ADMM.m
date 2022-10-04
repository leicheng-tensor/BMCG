% MC_SOLVE_ADMM     Use ADMM for matrix completion with graphs
%
% minimize_X   1/2 ||A(X) - m||^2    + gamma_n||X||_*
%                                    + gamma_c/2||grad_Gc(X)||_F^2
%                                    + gamma_r/2||grad_Gr(X)||_F^2      (1)
%
% where grad_Gr(X) is the gradient of X wrt the rows graph and grad_Gc(X)
% is the gradient wrt the columns graph.
%
% see [1] for details.
%
%
% Usage:
%   [X, stat] = MC_solve_ADMM(m_train, m_val, m_test, prob_par, solver_par)
%
% INPUTS:
%           m_train:  vector containing the train data
%           m_val:         >>               validation data
%           m_test:        >>               test data
%               ** essentially m_train = A_op(X_train(:))               **
%               ** the input data is AFTER POSSIBLE LINEAR RESCALING    **
%               ** EXPLAINED by solver_par.y_lims_init,                 **
%               ** solver_par.y_lims_scaled (see below)                 **
%           prob_par: Struct with parameters of the problem to be solved
%               with FIELDS:
%               gamma_n:    regularization constant for nuclear norm term
%               gamma_c:    regularization constant for columns graph
%               gamma_r:    regularization constant for rows graph
%               size_X: size of initial matrix X
%               Lc:         Laplacian of columns graph
%               Lr:         Laplacian of rows graph
%               A_op:   sampling operator A of problem (1). Should be
%                       applicable to the vector form of X
%               At_op:  adjoint of A
%               AtA_op: @(x) At_op(A_op(x)) or a faster implementation
%
%           solver_par: Struct with parameters for the solver with FIELDS:
%               tol_abs:    absolute size stopping criterion
%               tol_rel:    relative size stopping criterion
%               maxit:      maximum number of iterations
%               y_lims_init:    range of values of ORIGINAL X before
%                               rescaling. This is used because usually for
%                               nuclear norm methods the data has to be
%                               first rescaled to zero mean.
%               y_lims_scaled:  range of values AFTER RESCALING. This is
%                               the range of values of the input m_train,
%                               m_val and m_test.
%               rho_ADMM:   step size of ADMM algorithm
%               svds:   Use svds (1) or svd (0) for proximal of nuclear
%                       norm?
%               verbose:    Level of verbosity of algorithm
%                           0 = no output
%                           1 = print 
%                               [residual_primal,   tolerance_primal,...
%                                residual_dual,     tolerance_dual]
%                               per iteration. When BOTH residuals are
%                               smaller than the corresponding tolerances,
%                               the algorithm stops. Useful flag to set
%                               rho_ADMM accordingly!
%                           2 = additionaly print rank and nuclear norm of
%                               current solution.
%
%
%
% OUTPUTS:
%           X: final solution (full matrix!) in the INITIAL RANGE
%           stat: statistics of algorithm per iteration. FIELDS:
%               last_iter:  final iteration at convergence
%               f_obj: objective function
%               rankX: rank of matrix
%               mae_val:    mean absolute error     of validation data   
%               rmse_val:   root mean squared error         >>
%               mae_test:   mean absolute error     of test data
%               rmse_test:  root mean squared error         >>
%               mae_val_round,
%               rmse_val_round,
%               mae_test_round,
%               rmse_test_round:
%                           same as above but after PROJECTING TO [1,5] and
%                           ROUNDING to closest integer
%
%
% REFERENCES:
%   [1] V. Kalofolias, X. Bresson, M. Bronstein and P. Vandergheynst.
%   Matrix Completion on Graphs. Neural Information Processing Systems
%   2014, Workshop "Out of the Box: Robustness in High Dimension",
%   Montreal, Canada, 2014.
%
%see also   MC_demo_solve_ADMM, MC_solve_forward_backward, split_observed
%           average_error, lin_map, sample_sparse, sample_sparse_t,
%           sample_sparse_AtA
%
%code author: Vassilis Kalofolias
%date: Apr 2014

% note that the input m_train is basically A_op(vec(M_train)) where M_train
% is a matrix of the size of X.



function [X, stat] = MC_solve_ADMM(m_train, m_val, m_test, prob_params, solver_params)


%% Problem parameters
Lc = prob_params.Lc;
Lr = prob_params.Lr;

% the next are used instead of mask_train:
A_op = prob_params.A_op;
At_op = prob_params.At_op;          
AtA_op = prob_params.AtA_op;

% for masks of validation and test we don't use operators to keep it simple
if not(isempty(m_val)), mask_val = prob_params.mask_val; end;
if not(isempty(m_test)), mask_test = prob_params.mask_test; end;

% the regularization constants
gamma_n = prob_params.gamma_n;  % nuclear norm
gamma_r = prob_params.gamma_r;  % rows graph
gamma_c = prob_params.gamma_c;  % columns graph

size_X = prob_params.size_X;    % needed to go from m_train to M_train
numel_X = prod(size_X);               % numel(X)

%% PARAMETERS OF ALGORITHM:
TOL_ABS = solver_params.tol_abs;
TOL_REL = solver_params.tol_rel;

maxit = solver_params.maxit;
y_lims_init = solver_params.y_lims_init;
y_lims_scaled = solver_params.y_lims_scaled;
rho_ADMM = solver_params.rho_ADMM;

% for prox of nuclear norm
param_nuclear.svds = solver_params.svds;         % TODO: this should be input
param_nuclear.verbose = solver_params.verbose - 1;

% param_nuclear.svds = true;
% param_nuclear.max_rank = 20;
% param_nuclear.tol = 1e-3;


%% operators needed for STEP 2 of ADMM: (solve C(Y) = b)
C_op = @(y) AtA_op(y) + vec(gamma_r * Lr * reshape(y, size_X) + gamma_c * reshape(y, size_X) * Lc + rho_ADMM * reshape(y, size_X));
% to precondition Conjugate Gradients, we provide the inverse of the
% diagonal of the C operator matrix
%D_inv_op = @(y) y ./ (rho_ADMM + AtA_op(ones(size(y))) + gamma_r * kron(ones(size_X(2), 1), diag(Lr)) + gamma_c * kron(diag(Lc), ones(size_X(1), 1)));
AtAM = reshape(At_op(m_train), size_X);

%% ADMM Initialization
Z = AtAM;
if isa(m_train, 'single')
    Y = single(zeros(size_X)); % is this a valid starting point?
else
    Y = zeros(size_X);
end

Y_old = Y;

%% stats initialization
stat.f_obj = nan(maxit,1);
stat.rankX = nan(maxit,1);

stat.mae_val           = nan(maxit,1);
stat.rmse_val          = nan(maxit,1);
stat.mae_val_round     = nan(maxit,1);
stat.rmse_val_round    = nan(maxit,1);

stat.mae_test          = nan(maxit,1);
stat.rmse_test         = nan(maxit,1);
stat.mae_test_round    = nan(maxit,1);
stat.rmse_test_round   = nan(maxit,1);

%stat.res_pri  = nan(maxit,1);
%stat.res_dual = nan(maxit,1);
%stat.eps_pri  = nan(maxit,1);
%stat.eps_dual = nan(maxit,1);
res_pri  = nan(maxit,1);
res_dual = nan(maxit,1);
eps_pri  = nan(maxit,1);
eps_dual = nan(maxit,1);

pcg_tol = 1e-5;  % this should be input?

% X_best_mae = [];
% best_mae = 1;
% X_best_rmse = [];
% best_rmse = 1;

if solver_params.verbose > 0
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'primal grad', 'primal eps', 'dual grad', 'dual eps', 'objective');
end

%% ADMM:
for i = 1 : maxit
    
  %  fprintf('iter %d:  ', i);
    
    %% STEP 1: X <- argmin (gamma_n||X||_*  +  rho/2 ||X - (Y - Z)||_F^2)
    [X, info_nuclear] = prox_nuclearnorm(Y-Z, gamma_n/rho_ADMM, param_nuclear);
    param_nuclear.max_rank = round(max(5, info_nuclear.rank) * 1.5);
    
    %% STEP 2: Y <- argmin (    1/2 ||A(Y) - m||^2 
    %%                         + gamma_c/2 ||grad_Gc(Y)||_F^2
    %%                         + gamma_r/2 ||grad_Gr(Y)||_F^2 
    %%                         + rho/2    ||X - (Y - Z)||_F^2)
    % TODO: find a better way to solve this system !!!
    [y, ~, ~, ~, resvec] = pcg(C_op, vec(AtAM + rho_ADMM * (X + Z)), pcg_tol, 30, [], [], vec(X));
    %[y] = pcg(C_op, vec(AtAM + rho_ADMM * (X + Z)), [], 100 );
    
    Y = reshape(y, size_X);

    %% STEP 3: Z <- Z + X - Y
    Z = Z + X - Y;
    
            
    %% Compute statistics:
    % THE FOLLOWING WOULD HOLD BEFORE GOING BACK TO INITIAL SCALE
    % train_error(i) = norm(m_train - A_op(vec(X))) / norm(m_train);
    % validation_error(i) = norm(m_val - X(mask_validate)) / norm(m_val);

    % the overhead of the following is nothing compared to prox_nuclearnorm
    stat.f_obj(i) =  1/2 *           norm(m_train - A_op(vec(X)))^2 + ...
                gamma_c/2 *     sum(sum( (X*(Lc)) .* X  )) + ...
                gamma_r/2 *     sum(sum(  X .* ((Lr)*X) )) + ...
                gamma_n *       info_nuclear.final_eval;     % nuclear norm
    stat.rankX(i) = info_nuclear.rank;           % rank of current solution
    
    if not(isempty(m_val))
        % MAE and RMSE after linear mapping back to [1, 5]
        [stat.mae_val(i), stat.rmse_val(i)] = average_error(X(mask_val), m_val, y_lims_init, y_lims_scaled, 0);
        % MAE and RMSE after linear mapping to [1,5] AND rounding on {1, ...,5}
        [stat.mae_val_round(i), stat.rmse_val_round(i)] = average_error(X(mask_val), m_val, y_lims_init, y_lims_scaled, 1);
    end
    
    if not(isempty(m_test))
        % MAE and RMSE after linear mapping back to [1, 5]
        [stat.mae_test(i), stat.rmse_test(i)] = average_error(X(mask_test), m_test, y_lims_init, y_lims_scaled, 0);
        % MAE and RMSE after linear mapping to [1,5] AND rounding on {1, ...,5}
        [stat.mae_test_round(i), stat.rmse_test_round(i)] = average_error(X(mask_test), m_test, y_lims_init, y_lims_scaled, 1);
    end        
        
    %% STOPPING CRITERION:
    %stat.res_pri(i) = norm(X - Y, 'fro');
    %stat.res_dual(i) = rho_ADMM * norm(Y_old - Y, 'fro');
    res_pri(i) = norm(X - Y, 'fro');
    res_dual(i) = rho_ADMM * norm(Y_old - Y, 'fro');
    
    %stat.eps_pri(i)  = sqrt(numel_X) * TOL_ABS + TOL_REL * max(norm(X, 'fro'), norm(Y, 'fro'));
    %stat.eps_dual(i) = sqrt(numel_X) * TOL_ABS + TOL_REL * norm(Z, 'fro');
    eps_pri(i)  = sqrt(numel_X) * TOL_ABS + TOL_REL * max(norm(X, 'fro'), norm(Y, 'fro'));
    eps_dual(i) = sqrt(numel_X) * TOL_ABS + TOL_REL * norm(Z, 'fro');
    
    if solver_params.verbose > 0
%        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
%            stat.res_pri(i), stat.eps_pri(i), ...
%            stat.res_dual(i), stat.eps_dual(i), stat.f_obj(i));
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
            res_pri(i), eps_pri(i), ...
            res_dual(i), eps_dual(i), stat.f_obj(i));
    end
    
    %if stat.res_pri(i) < stat.eps_pri(i) && stat.res_dual(i) < stat.eps_dual(i)
    if res_pri(i) < eps_pri(i) && res_dual(i) < eps_dual(i)
        break
    end
    
    Y_old = Y;
    
end

stat.last_iter = i;

X = lin_map(X, y_lims_init, y_lims_scaled);

end
