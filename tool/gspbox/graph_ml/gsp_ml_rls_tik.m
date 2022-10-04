function [sol, infos] = gsp_ml_rls_tik(G, xl, y, k, tau,lambda, A, At, param)
%GSP_ML_RLS_TIK Manifold Learning regularized least square with laplacian regularization
%   Usage: sol = gsp_ml_rls_tik(G, xl, y, k, tau, lambda);
%          sol = gsp_ml_rls_tik(G, xl, y, k, tau, lambda, A, At);
%          sol = gsp_ml_rls_tik(G, xl, y, k, tau, lambda, A, At, param);
%          [sol, infos] = gsp_ml_rls_tik(...)
%   
%   Input parameters:
%       G       : Gsp graph
%       xl      : labeled points
%       y       : labels
%       k       : kernel
%       tau     : regularization parameters
%       lambda  : regularization parameters
%       A       : Operator
%       At      : Adoint operator
%       param   : Optional parameters
%   Output parameters:
%       sol     : solution of the problem (kernel coefficients)
%       infos   : convergence info
%       
%   param is a structure of optional argument given to the solver
%   gradient_descent. Please see the function gradient descent for more
%   information. 
%
%   In param, you also have to set an upperbound for the operator A as
%   param.nu!
%
%   This function solves the following problem:
%
%       argmin_alpha  || A (K alpha) - y ||_2^2 
%                       + tau *alpha^T K alpha 
%                       + lambda alpha^T K^T L K alpha
%
%   If tau is set to zero, then the following problem is solved
%
%       argmin_alpha alpha^T K alpha
%                      + lambda alpha^T K^T L K alpha  
%                      s. t.  A (K alpha) = y
%

% Author: Nathanael Perraudin
% Date  : 8 decembre 2014


if nargin<7
%
%   Url: https://epfl-lts2.github.io/gspbox-html/doc/graph_ml/gsp_ml_rls_tik.html

% Copyright (C) 2013-2016 Nathanael Perraudin, Johan Paratte, David I Shuman.
% This file is part of GSPbox version 0.7.5
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% If you use this toolbox please kindly cite
%     N. Perraudin, J. Paratte, D. Shuman, V. Kalofolias, P. Vandergheynst,
%     and D. K. Hammond. GSPBOX: A toolbox for signal processing on graphs.
%     ArXiv e-prints, Aug. 2014.
% http://arxiv.org/abs/1408.5781
    A = @(x) x;
end


if nargin<8
    At = A;
end

if nargin<9
    param = struct;
end


if ~isfield(param, 'tol'), param.tol = 1e-6; end
if ~isfield(param, 'nu'), param.nu = 1; end
if ~isfield(param, 'verbose'), param.verbose = 1; end
    
    

N = size(xl,2);

% Evaluate the kernel on the data points
K = gsp_rkhs_evaluate(k,xl);
nu = norm(K);

alpha_in = zeros(N,size(y,2));

if ~isfield(G,'lmax');
    G = gsp_estimate_lmax(G);
end
    
flap.eval = @(x) lambda*sum(gsp_norm_tik(G,K*x));
flap.grad = @(x) 2*lambda*full(G.L*(K*x));
flap.beta = 2*lambda*nu*G.lmax;

if tau >0
    
    fp.eval = @(x) tau *sum(gsp_norm_tik(G,x));
    fp.grad = @(x) 2*tau*K*x;
    fp.beta = 2* tau*nu+nu^2;

    ffid.eval = @(x) norm(A(K*x)-y,'fro')^2;
    ffid.grad = @(x) 2*K'*At(A(K*x)-y);
    ffid.beta = param.nu^2;
    

    [sol,infos] = solvep(alpha_in, {flap,ffid,fp}, param);
else
    
    fp.eval = @(x)  x'*K*x;
    fp.grad = @(x) 2*K*x;
    
    ftot.grad = @(x)  fp.grad(x) + flap.grad(x);
    ftot.eval = @(x)  fp.eval(x) + flap.eval(x);
    
    paramproj.A = @(x) A(K*x);
    paramproj.At = @(x) K'*At(x);
    paramproj.nu = nu^2*param.nu^2;
    paramproj.tight = 0;
    paramproj.verbose = param.verbose-1;
    paramproj.y = y';
    paramproj.maxit = 50;
    ffid.eval = @(x) eps;
    ffid.prox = @(x,T) proj_b2(x,T,paramproj);

    param.gamma = 0.5/(nu+lambda*nu*G.lmax);

    [sol,infos] = forward_backward(alpha_in, ffid, ftot, param);

end




end
