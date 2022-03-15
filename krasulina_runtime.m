clc
% load scene

data = X;
[N, dim] = size(X);
data = data/max(data(:));
data = bsxfun(@minus, data, mean(data,1));


Y = data';
idx = randperm(size(Y,2));
Y = Y(:,idx);

k_values = [1, 2, 5, 10, 20, 50, 80, 100, 150, 200];
num_iters = 10;
runtime_krasulina = zeros(length(k_values), num_iters);
eta0 = 0.1;
alpha = 0.5;

for iter = 1:num_iters
    for kk = 1:length(k_values)
        k = k_values(kk);
        C = 0.1 * randn(size(Y,1),k);
        Cinv = inv(C' * C);
        tic
        for ii = 1:size(Y,2)
            y = Y(:,ii);
            eta = eta0/(ii^alpha);
            mu = Cinv * (C' * y);
            yc = C'* y;
            eta = eta/(1 + eta * (mu' * mu));
            C = C + eta * (y - C * mu) * mu';
            U = [mu, eta * yc];
            V = [eta * yc + eta^2 * (y' * y) * mu, mu];
            Cinv = Cinv - Cinv * U/(eye(2) + V' * Cinv * U) * (V' * Cinv); 
        end
        t = toc;
        runtime_krasulina(kk, iter) = t;
        fprintf('iter = %d, k = %d\n', iter, kk)
    end
end