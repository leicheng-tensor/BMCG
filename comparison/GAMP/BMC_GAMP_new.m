function Result = BMC_GAMP_new(Y,Omega,Max_iter,W1_inv)
    [M,N] = size(Y);
    if N<M
        Y = Y.';
        [M,N] = size(Y);
        Trp_flag = 1;
        Omega = Omega.';
    else
        Trp_flag = 0;
    end
    X = Y+0.5*~Omega;
    Sigma = ((X*X')/N+eye(M));
    gamma = 1;
    NN = nnz(Omega);
    a = 1e-10;
    c = 1e-10;
    d = 1e-10;
    Me = Y.*Omega;
    for iter = 1 : Max_iter
        Max_count = 1;
        for count = 1 : Max_count
            X_old = X;
            Sigma_inv = eye(M)/(Sigma+(1e-10)*eye(M));
            % Update X via GAMP
            for ii = 1 : 1
    %             X_old = X;
                Tao_r = repmat(1./(diag(Sigma_inv)),1,N);
                R = X - Tao_r.*(Sigma_inv*X); 
                gamma_temp = gamma.*Omega + 0.001.*~Omega;
                Tao_x = Tao_r./(1 + Tao_r.*gamma_temp);
                X = Tao_x.*(R./Tao_r+Me.*gamma);
            end
            Sigma = (X*X' + diag(sum(Tao_x,2))+W1_inv)/(N+a);
            % Update gamma
            gamma = (0.5*NN+c)/(0.5*sum(sum((abs(X-Me).^2 + Tao_x).*Omega))+d);
        end
        if norm(X-X_old,'fro')/norm(X,'fro')<1e-8
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

