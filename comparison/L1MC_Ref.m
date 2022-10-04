function Result = L1MC_Ref(Y, Omega,R,mu,Max_iter)
%--------------------------------------------------------------------------
% Author: Linxiao Yang
% Date : 2017/12/29
% Intro:
%--------------------------------------------------------------------------
    R_e = L1MC(Y,Omega,R,mu,Max_iter);
    X_e = R1MC(Y,Omega,R_e,Max_iter);
    Result.X = X_e;
    Result.r = R_e;
end
function R_e = L1MC(Y,Omega,R,mu,Max_iter)
    [M,N] = size(Y);
    X = Y.*Omega;
    w = rand(R,1);
    U = randn(M,R);
    V = randn(N,R);
    SR = nnz(Omega)/M/N;
    for ii = 1 : R
        U(:,ii) = U(:,ii)/norm(U(:,ii));
        V(:,ii) = V(:,ii)/norm(V(:,ii));
    end
    for iter = 1 : Max_iter
        X_old = X;
        X_r = X;
        for ii = 1 : R
            if w(ii) ~= 0
                ut = X_r*V(:,ii);
                U(:,ii) = ut/norm(ut)*sign(w(ii));
                vt = X_r'*U(:,ii);
                V(:,ii) = vt/(norm(vt))*sign(w(ii));
                w(ii) = max(0,U(:,ii)'*X_r*V(:,ii)-mu)+min(0,U(:,ii)'*X_r*V(:,ii)+mu);
                X_r = X_r - w(ii)*U(:,ii)*V(:,ii)';
            end
        end
        Z = X-X_r;
        X = Y.*Omega+Z.*~Omega;
        if min(norm((X-Z).*Omega,'fro')/norm(X.*Omega,'fro'),norm(X-X_old,'fro')/norm(X,'fro'))<1e-6
            break;
        end
    end
    R_e = nnz(w>(1e-3)*SR*sum(w));
end
function X_e = R1MC(Y,Omega,R_e,Max_iter)
    X = Y.*Omega;
    for iter = 1 : Max_iter
        X_old = X;
        [U,S,V] = svd(X,'econ');
        Z = U(:,1:R_e)*S(1:R_e,1:R_e)*V(:,1:R_e)';
        X = Y.*Omega + Z.*~Omega;
        if min(norm((X-Z).*Omega,'fro')/norm(X.*Omega,'fro'),norm(X-X_old,'fro')/norm(X,'fro'))<1e-6
            break;
        end
    end
    X_e = X;
end




































