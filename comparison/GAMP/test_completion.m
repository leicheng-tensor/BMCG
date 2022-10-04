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
            W1_inv = eye(min(M,N))*(1e-10);
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
            %% Recover with BiGAMP
            fprintf('Run BiGAMP\n');
            % Generate the Unknown Low Rank Matrix
            Z = Y;
            error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
            nuw = 0;
            Y(~Omega) = 0;
            %Set options
            opt = BiGAMPOpt; %initialize the options object with defaults
            %Use sparse mode for low sampling rates
            if p <= 0.2
                opt.sparseMode = 1;
            end
            %Provide BiG-AMP the error function for plotting NMSE
            opt.error_function = error_function;
            problem = BiGAMPProblem();
            problem.M = M;
            problem.N = [];
            problem.L = N;
            results = [];
            [problem.rowLocations,problem.columnLocations] = find(Omega);
            gX = AwgnEstimIn(0, 1);
            %Prior on A is also Gaussian
            gA = AwgnEstimIn(0, 1);
            %Log likelihood is Gaussian, i.e. we are considering AWGN
            if opt.sparseMode
                %In sparse mode, only the observed entries are stored
                gOut = AwgnEstimOut(reshape(Y(Omega),1,[]), nuw);
            else
                gOut = AwgnEstimOut(Y, nuw);
            end
            %We limit the number of iterations that BiG-AMP is allowed during each
            %EM iteration to reduce run time
            opt.nit = 250; %limit iterations

            %We also override a few of the default EM options. Options that we do not
            %specify will use their defaults
            EMopt.maxEMiter = 200; %This is the total number of EM iterations allowed
            EMopt.maxEMiterInner = 5; %This is the number of EM iterations allowed for each rank guess
            EMopt.learnRank = 1; %The AICc method is option 1
            %Run BGAMP
            tic
            [estFin,~,~,estHist] = EMBiGAMP_MC(Y,problem,opt,EMopt);
            toc
            tt4(pp,r) = tt4(pp,r) + toc;
            err4 = norm(estFin.Ahat*estFin.xhat - X,'fro')/norm(X,'fro');
            if err4 < 1e-2
                success_rate4(pp,r) = success_rate4(pp,r) + 1;
            end
            %% Recover with alm_mc
            fprintf('Run alm_mc\n');
            tic
            [A, iter, svp] = inexact_alm_mc(Y.*Omega, -1, 1000);
            toc
            tt5(pp,r) = tt5(pp,r) + toc;
            err5 = norm(A.U*A.V' - X,'fro')/norm(X,'fro');
            if err5 < 1e-2
                success_rate5(pp,r) = success_rate5(pp,r) + 1;
            end
            %% Recover with L1MC
            fprintf('Run L1MC\n');
            Max_iter = 1000;
            tic
            Result2 = L1MC_Ref(Y, Omega,fix(rank*nn),10,Max_iter);
            toc
            tt6(pp,r) = tt6(pp,r) + toc;
            err6 = norm(Result2.X - X,'fro')/norm(X,'fro');
            if err6 < 1e-2
                success_rate6(pp,r) = success_rate6(pp,r) + 1;
            end
        end
    end
end
