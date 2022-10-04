clear;clc;
M = 500;                                    %row of matrix
N = 500;                                     %column of matrix
p_set = 0.2;%[0.2,0.5];
r_set = 20:5:130;%40 : 5 : 70;
success_rate1 = zeros(length(p_set),length(r_set));
success_rate2 = zeros(length(p_set),length(r_set));
success_rate3 = zeros(length(p_set),length(r_set));
success_rate4 = zeros(length(p_set),length(r_set));
success_rate5 = zeros(length(p_set),length(r_set));
success_rate6 = zeros(length(p_set),length(r_set));
tt1 = zeros(length(p_set),length(r_set));
tt2 = zeros(length(p_set),length(r_set));
tt3 = zeros(length(p_set),length(r_set));
tt4 = zeros(length(p_set),length(r_set));
tt5 = zeros(length(p_set),length(r_set));
tt6 = zeros(length(p_set),length(r_set));
%Add paths for matrix completion
basePath = [fileparts(mfilename('fullpath')) filesep];
%GAMPMATLAB paths
addpath([basePath 'BiGAMP']) %BiG-AMP code
addpath([basePath 'main']) %main GAMPMATLAB code
addpath([basePath 'EMGMAMP']) %EMGMAMP code
addpath([basePath 'PROPACK']) %EMGMAMP code
nn = 2;
for pp = 1 : length(p_set)
    p = p_set(pp);
    for r = 1 : length(r_set)
        rank = r_set(r);
        for count = 1 : 25
            [p,rank,count]
            %% Form low-rank matrix
            X = randn(M,rank)*randn(rank,N);
            Omega = zeros(M,N);
            Omega(randperm(M*N,round(M*N*p)))=1;
            Omega = logical(Omega);
            Y = (X+0*0.1*randn(M,N)).*Omega;
            %% Recover with BMC_GAMP
            fprintf('Run BMC_GAMP\n');
            Max_iter = 2000;
            W1_inv = zeros(M); 
            for ii = 1 : M
                for jj = 1 : M
                    W1_inv(ii,jj) = exp(-(ii-jj)^2/3);
                end
            end
            W1_inv = diag(sum(W1_inv)) - W1_inv;
            W1_inv = eye(M)/(W1_inv+eye(M)*(1e-10));
            tic
            Result1 = BMC_GAMP(Y,Omega,Max_iter,W1_inv);
            toc
            tt1(pp,r) = tt1(pp,r) + toc;
            err1 = norm(Result1.X - X,'fro')/norm(X,'fro');
            if err1 < 1e-2
                success_rate1(pp,r) = success_rate1(pp,r) + 1;
            end
            %% Recover with VBMC
            fprintf('Run VBMC\n');
            options.MAXITER = 100; 
            options.DIMRED = 1; % Reduce dimensionality during iterations?
            options.UPDATE_BETA = 1;
            options.initial_rank = fix(rank*nn); % or we can set a value. 
            options.verbose = 0;
            P = Omega;
            tic
            [X_hat, A_hat, B_hat] = VBMC(P, Y, options);
            toc
            tt2(pp,r) = tt2(pp,r) + toc;
            err2 = norm(X_hat - X,'fro')/norm(X,'fro');
            if err2 < 1e-2
                success_rate2(pp,r) = success_rate2(pp,r) + 1;
            end
            %% Recover with LMaFit
            fprintf('Run LMaFit\n');
            % problem specification
            opts = [];
            TT = (1 : M*N)';
            Known = TT(Omega);
            data = Y(Omega);
            % call solver
            tic; 
            [Xt,Yt,Out] = lmafit_mc_adp(M,N,fix(rank*nn),Known,data,opts);
            toc
            tt3(pp,r) = tt3(pp,r) + toc;
            err3 = norm(Xt*Yt - X,'fro')/norm(X,'fro');
            if err3 < 1e-2
                success_rate3(pp,r) = success_rate3(pp,r) + 1;
            end
        end
    end
end
