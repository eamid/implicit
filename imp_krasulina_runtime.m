load mnist
load centered_pca_results_mnist.mat

N = size(X,1);
X = double(X);
X = X/max(X(:));
X = bsxfun(@minus, X, mean(X,1));

Y = X(randperm(size(X,1)),:)';
Y_val = Y(:,1:round(0.1 * size(X,1)));

alpha = 0.8;

k_values = [5, 10, 20, 50];


for kk = 1
    k = k_values(kk);
    C_init = 0.00001 * randn(size(Y,1),k);
%     eta_vals = [0.01, 0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 100.0 * k, 1000.0 * k];
    eta_vals = [1000.0, 10000.0, 100000.0];
    error_val = zeros(1, length(eta_vals));
    for ee = 1:length(eta_vals)
        eta0 = eta_vals(ee);
        C = C_init;
%         Cinv = (C' * C)\C';
        Cinv = inv(C' * C);
        for ii = 1:size(Y_val,2)
            y = Y_val(:,ii);
            eta = eta0/(ii^alpha);
            yc = C' * y;
            mu = Cinv * yc;
            eta = eta/(1 + eta * (mu' * mu));
            r = eta * (y - C * mu);
            C = C +  r * mu';
            cr = C' * r;
            U = [mu, cr];
            V = [eta * cr + (r' * r) * mu, eta * mu];
            if mod(ii,1000) == 0
                Cinv = inv(C' * C);
            else
                Cinv = Cinv - Cinv * U/(eye(2) + V' * Cinv * U) * (V' * Cinv);
            end
            
%             Cinv = OneRankInverseUpdate(C, Cinv, eta * (y - C * mu), mu);
        end
        Xh = C\Y_val;
        error_val(ee) = mean(sum((Y_val - C * Xh).^2, 1));
    end
    [~,idx_eta] = min(error_val);
    eta0_best = eta_vals(idx_eta);
    
    for iter = 1:10
        Y = X(randperm(size(X,1)),:)';
        cc = 1;
        elapsed = 0;
        t = cputime;
        C = C_init;
        Cinv = inv(C' * C);
%         Cinv = (C' * C)\C';
%         t = toc;
        elapsed = elapsed + cputime - t;
        Xh = C\Y;
        error(4,kk,cc,iter) = mean(sum((Y - C * Xh).^2, 1));
        runtime(4,kk,cc,iter) = elapsed;
        cc = cc + 1;
        for ii = 1:size(Y,2)
%             y = Y(:,ii);
%             t  = cputime;
%             eta = eta0_best/(ii^alpha);
%             mu = C\y;
%             eta = eta/(1 + eta * (mu' * mu));
%             C = C + eta * (y - C * mu) * mu';
%             Cinv = OneRankInverseUpdate(C, Cinv, eta * (y - C * mu), mu);
            y = Y(:,ii);
            t  = cputime;
            eta = eta0_best/(ii^alpha);
            yc = C'* y;
            mu = Cinv * yc;
            eta = eta/(1 + eta * (mu' * mu));
            r = eta * (y - C * mu);
            C = C +  r * mu';
            if mod(ii,1000) == 0
                Cinv = inv(C' * C);
            else
                cr = C' * r;
                U = [mu, cr];
                V = [eta * cr + (r' * r) * mu, eta * mu];
                Cinv = Cinv - Cinv * U/(eye(2) + V' * Cinv * U) * (V' * Cinv);
            end
            elapsed = elapsed + cputime - t;
            if mod(ii, 5000) == 0
                Xh = C\Y;
                error(4,kk,cc,iter) = mean(sum((Y - C * Xh).^2, 1));
                runtime(4,kk,cc,iter) = elapsed;
                cc = cc + 1;
            end
        end
        fprintf("k %d, iter %d, error %f \n", k, iter, error(4,kk,cc-1,iter));
        save('centered_pca_results_mnist.mat', 'error', 'runtime')
    end
end
    