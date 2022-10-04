function Result = BMC_GAMP(Y,Omega,Max_iter,W1_inv)
    [M,N] = size(Y);
    if N<M
        Y = Y.';
        [M,N] = size(Y);
        Trp_flag = 1;
        Omega = Omega.';
    else
        Trp_flag = 0;
    end
    X = Y;
    Sigma_x = eye(M);
    Sigma = ((X*X')/N+Sigma_x);
    gamma = 1;
    NN = nnz(Omega);
    a = 1e-10;
    b = 1e-10;
    c = 1e-10;
    d = 1e-10;
    Me = Y.*Omega;
    for iter = 1 : Max_iter
        X_old = X;
        [U,W] = eig(Sigma);
        W = repmat(diag(W),1,N);
        U2 = abs(U').^2;
        S = zeros(size(W));
        Tao_x = 0.1*ones(size(X))/sqrt(iter);
        % Update X via GAMP
        for ii = 1 : 1
%             X_old = X;
            Z = U'*X;
            Tao_p = U2*Tao_x;
            P = Z - Tao_p.*S;
            Tao_z = (Tao_p.*W)./(Tao_p+W);
            Z0 = Tao_z.*(P./Tao_p);
            S = (Z0 - P)./Tao_p;
            Tao_s = (Tao_p - Tao_z)./(Tao_p.^2);
            Tao_r = 1./(U2'*Tao_s);
            R = X + Tao_r.*(U*S); 
            gamma_temp = gamma.*Omega + 0.001.*~Omega;
            Tao_x = Tao_r./(1 + Tao_r.*gamma_temp);
            X = Tao_x.*(R./Tao_r+Me.*gamma);
        end
        Sigma = (X*X' + diag(sum(Tao_x,2))+W1_inv)/(N+a);
        % Update gamma
        gamma = (0.5*NN+c)/(0.5*sum(sum((abs(X-Me).^2 + Tao_x).*Omega))+d);
        if norm(X-X_old,'fro')/norm(X,'fro')<1e-5
            break;
        end
    end
    if Trp_flag == 1
        Result.X = X';
        Result.Sigma = Sigma;
        Result.gamma = gamma;
    else
        Result.X = X;
        Result.Sigma = Sigma;
        Result.gamma = gamma;
    end
end
