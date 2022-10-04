function[Result, stat] = BMCG_update(M, varargin)
%   Bayesian Matrix Completion on Dual Graphs
%
%   [Result, stat] = BMCG(M, 'param1', val1, 'param2', val2, ...)
%
%   Inputs
%       M:  Input observed matrix, m*n
%       'obs': Binary observation matrix of same size as M
%              (0: MISSING 1: OBSERVED)
%       'initRank': The initialization of rank (larger than true rank),
%                   k<=min(m,n)
%       'maxiters': max number of iterations
%       'tol': lower band change tolerance for convergence dection
%       'Gu': Laplacian of factor matrix U
%       'Gv': Laplacian of factor matrix V
%
%   Outputs
%       Result: final solution X = UV'(full matrix); estimated Rank k
%       stat: statistics of algorithm per iteration
%             ***tbc***
%
%   Example
%       [X, stat] = BMCG(M, 'obs', Omega, 'initRank', max[dim, 2*R],
%       'maxiters', 100, 'tol', 1e-6, 'Gu', Lu, 'Gv', Lv)

%% Set parameters
[m,n] = size(M);

ip = inputParser;
ip.addParameter('obs', ones(m,n),@(x)(isnumeric(x)||islogical(x)));
ip.addParameter('V', zeros(m,n),@(x)(isnumeric(x)||islogical(x)));
ip.addParameter('initRank', min(m,n), @isscalar); %%%%!!!ATTENTION
ip.addParameter('maxiters', 100, @isscalar);
%ip.addParameter('tol', 1e-5, @isscalar);
ip.addParameter('Gu', eye(m), @(x)(isnumeric(x)||islogical(x)));
ip.addParameter('Gv', eye(n), @(x)(isnumeric(x)||islogical(x)));
ip.addParameter('val', zeros(m,n),@(x)(isnumeric(x)||islogical(x)));
ip.parse(varargin{:});

O = ip.Results.obs;
k = ip.Results.initRank;
maxiters = ip.Results.maxiters;
%tol = ip.Results.tol;
Lu = ip.Results.Gu;
Lv = ip.Results.Gv;
Val = ip.Results.val;
mask_val = ip.Results.V;
%% Initialization
% M = M.*O;
Nobs = sum(O(:));
% for noise
a0 = 1e-6;
b0 = 1e-6;
tao = 1;

% for covariance
c0 = 1e-6; % noisy case
d0 = 1e-6;
lambdas = ones(k,1);

% for factors X = abc' = UV'(svd init)
% dscale of M from BCPF_TC
dscale = sum((M(:)-sum(M(:)/Nobs)).^2)/Nobs;
dscale = sqrt(dscale)/2;
M = M./dscale;  
USigma = cell(1,k);
VSigma = cell(1,k);
[USigma{:}] = deal(eye(m));
[VSigma{:}] = deal(eye(n));
% X = M;
% X(O==0) = sum(M(:))/Nobs; % for factor initialization only
% [a,b,c] = svd(double(X), 'econ');
% U = a(:,1:k)*(b(1:k,1:k)).^(0.5);
% V = (b(1:k,1:k)).^(0.5)*c(:,1:k)';
% V = V';
X = (sum(M(:))/Nobs)*ones(m,n);
X = ~O.*X + M; % for factor initialization only
[a,b,c] = svds(X,k);
U = a*(b.^(0.5));
V = (b.^(0.5))*c';
V = V';
% old_rmse = 0;
% old_perct_recovery = get_perct(M0, dscale.*U*V', idx_unknown);
    %RMSE=full(sqrt(sum(sum((Val-dscale.*X).^2.*mask_val))/nnz(Val)));
% fprintf('init_perct_recovery = %g', old_perct_recovery); 
%M = M.*O;
%% Create figures
% tbc

%% Model learning
fprintf('\n----------Learning Begin----------\n')
for it = 1:maxiters
    %% update factor matrices U and V (column by column u1, u2..., v1, v2...)
    for i = 1:k
        EUrVr = U*V' - U(:,i)*V(:,i)';
        USigma{i} = inv(tao*diag(O*(V(:,i).*V(:,i)+diag(VSigma{i}))) + lambdas(i)*Lu);
        U(:,i) = tao*USigma{i}*(O.*(M - EUrVr))*V(:,i);
    end
    for i = 1:k
        EUrVr = U*V' - U(:,i)*V(:,i)';
        VSigma{i} = inv(tao*diag( O'*( (U(:,i).*U(:,i)+diag(USigma{i}) ))) + lambdas(i)*Lv);
        V(:,i) = tao*VSigma{i}*(O.*(M - EUrVr))'* U(:,i);
    end
    %% update latent matrix
    X = U*V';
    %% update hyperparameter lambda
    ck = (0.5*(m+n) + c0)*ones(k,1);
    dk = zeros(k,1);
    for r = 1:k
       dk(r) = d0 + 0.5.*(U(:,r)'*Lu*U(:,r) + trace(Lu*USigma{r}) + V(:,r)'*Lv*V(:,r)+ trace(Lv*VSigma{r}));
    end
    lambdas = ck./dk;
    %% update the noise tao
    ak = a0 + 0.5* Nobs;
    err = norm((O.*(M-X)),'fro').^2;
    for r=1:k
        USigmar = diag(USigma{r});
        VSigmar = diag(VSigma{r});
        for i = 1:m
            for j = 1:n
            if O(i,j)
            err = err + U(i,r).^2.*VSigmar(j) ...
                + V(j,r).^2.*USigmar(i) ...
                + USigmar(i)*VSigmar(j);
            end
            end
        end
    end
    bk = b0 + 0.5*err;
    tao = ak/bk;
    %% prune out unnecessary components 
    if it>0
        F = {U;V};
        F = cell2mat(F);
        Power = diag(F'*F);
        Pbound = max(Power)/1000;
        indices = (Power>Pbound);
        rank = sum(indices);
        U = U(:,indices);
        V = V(:,indices);
        USigma = USigma(indices);
        VSigma = VSigma(indices);
        lambdas = lambdas(indices);
        k = rank;
    end
    %% display progress
%     perct_recovery = get_perct(M0, dscale.*X, idx_unknown);
    rmse=full(sqrt(sum(sum((Val-dscale.*X).^2.*mask_val))/nnz(Val)));
%     if (it>5 && (old_rmse - rmse)<1e-6)
%         break;
%     end
    fprintf('Iter. %d: rmse = %g, Rank = %d \n', it, rmse, k); 
%     old_rmse = rmse;
end

%% prepare the results
X = U*V';
X = X*dscale;
% err = norm((O.*(M-X)),'fro').^2;
% Fit = 1-sqrt(sum(err(:)))/norm(M(:));
% SNR = 10*log10(var(X(:))*tao);
%% results
Result.X = X;
% Result.SNR = SNR;
% stat.Fit = Fit;
Result.EstRank = k;
stat.lambdas = lambdas;
stat.tao = tao;
stat.U = U;
stat.V = V;
stat.USigma = USigma;
stat.VSigma = VSigma;