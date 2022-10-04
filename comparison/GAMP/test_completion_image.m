clear;clc;                                    %column of matrix
X = double(imread('Lena512.tif'))/255;
[M,N] = size(X);
Y = X;
p = 0.3;
Omega = rand(M,N)<p;
Y = Y.*Omega;
P = sum(sum(Omega));
%%
W1 = eye(M);
W1(1,1) = 0.5;
W1(end,end) = 0.5;
for ii = 1 : M-1
    W1(ii,ii+1)=-0.5;
    W1(ii+1,ii)=-0.5;
end
W1_inv = eye(M)/(W1+(1e-10)*eye(M));
% W1_inv = eye(M)*(1e-10);
Max_iter = 100;
tic
Result1 = BMC_GAMP_new(Y,Omega,Max_iter,W1_inv);
toc
[psnr(Result1.X,X),ssim(Result1.X,X)]
%%
W1_inv = zeros(M); 
for ii = 1 : M
    for jj = 1 : M
        W1_inv(ii,jj) = exp(-(ii-jj)^2/3);
    end
end
W1_inv = diag(sum(W1_inv)) - W1_inv;
W1_inv = eye(M)/(W1_inv+eye(M)*(1e-10));
Max_iter = 100;
tic
Result2 = BMC_GAMP_new(Y,Omega,Max_iter,W1_inv);
toc
[psnr(Result2.X,X),ssim(Result2.X,X)]
%%
W1_inv = eye(M)*(1e-10);
Max_iter = 100;
tic
Result3 = BMC_GAMP_new(Y,Omega,Max_iter,W1_inv);
toc
[psnr(Result3.X,X),ssim(Result3.X,X)]
%% Run VBMC
% all options are *optional*, everything will be set automatically
% you can modify these options to get better performance
options.verbose = 1;
options.MAXITER = 200; 
options.DIMRED = 1; % Reduce dimensionality during iterations?
% you can also set the threshold to reduce dimensions
options.DIMRED_THR = 1e3;
%Estimate noise variance? (beta is inverse noise variance)
options.UPDATE_BETA = 1;
options.initial_rank = 120; % or we can set a value. 
options.X_true = Y; 
fprintf('Running VB matrix completion...\n');
tic
[X_hat, A_hat, B_hat] = VBMC(Omega, Y.*Omega, options);
t_total = toc;
[psnr(X_hat,X),ssim(X_hat,X)]
%%
fprintf('Run alm_mc\n');
tic
Result111 = ALM(Y.*Omega,Omega,1000);
toc
[psnr(Result111.X,X),ssim(Result111.X,X)]
%% Clean Slate

%Add paths for matrix completion
basePath = [fileparts(mfilename('fullpath')) filesep];

%GAMPMATLAB paths
addpath([basePath 'BiGAMP']) %BiG-AMP code
addpath([basePath 'main']) %main GAMPMATLAB code
addpath([basePath 'EMGMAMP']) %EMGMAMP code
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
%[psnr(estFin.Ahat*estFin.xhat,X),ssim(estFin.Ahat*estFin.xhat,X)]
%%
fprintf('Run LMaFit\n');
% problem specification
opts = [];
TT = (1 : M*N)';
Known = TT(Omega);
data = Y(Omega);
% call solver
tic; 
[Xt,Yt,Out] = lmafit_mc_adp(M,N,120,Known,data,opts);
toc
%[psnr(Xt*Yt,X),ssim(Xt*Yt,X)]
%%
Max_iter = 1000;
tic
Result5 = L1MC_Ref(Y, Omega,120,2,Max_iter);
toc
%[psnr(Result5.X,X),ssim(Result5.X,X)]
