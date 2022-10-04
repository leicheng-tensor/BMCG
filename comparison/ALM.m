function Result = ALM(Y,Omega,Max_iter)
%--------------------------------------------------------------------------
% Author: Linxiao Yang
% Date: 2017/7/18
% Intro:
%--------------------------------------------------------------------------
    [M,N] = size(Y);
    X = Y;
    D = zeros(M,N);
    Lambda = zeros(M,N);
    rho = 1e-3;
    for iter = 1 : Max_iter
        X_old = X;
        % Update X
        X = prox_nuclear(Y-D+Lambda/rho,1/rho);
        % Update D
        D = (Y+Lambda/rho-X).*~Omega;
        % Update Lambda
        Lambda = Lambda + rho*(Y-D-X);
        % Update rho
        rho = rho*1.2;
        if norm(X-X_old,'fro')/norm(X,'fro')<1e-8
            break;
        end
%         fprintf('Iteration:%d,NMSE=%f\n',iter,norm(X-X_real,'fro')/norm(X_real,'fro'))
    end
    Result.X = X;
end
function A = prox_nuclear(Q,tao)
    [U,S,V] = svd(Q,'econ');
    S = max(S-tao,0);
%     S(S>1e-3) = 1;
    A = U*S*V';
end