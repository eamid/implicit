clc
load mnist
k = 10;

data = X;
data = data/max(data(:));
m_data = mean(data,1);
data_pca = bsxfun(@minus, data, m_data);
tic
[coeff, S] = pca(data_pca, 'NumComponents', k);
toc
err_pca = mean(sum((data_pca - S * coeff').^2, 2));
disp(['PCA error ' num2str(err_pca)]);
%%
clc
Y = data_pca';
Y = Y(:, randperm(size(Y,2)));

err_iters = 5000:5000:70000;
errors = zeros(3, length(err_iters) + 1, 10);
eta0_vals = [1e4, 1e5, 1e6];
for i = 1:3
    eta0 = eta0_vals(i);
    for iter = 1:10
        C = 0.00001 * randn(size(Y,1),k);
        % eta = eta0;
        alpha = 0.8;
        idx = randperm(size(Y,2));
    %     tic
        cc = 1;
        beta = (C' * C)\C';
        X = beta * Y;
        err = mean(sum((Y - C * X).^2, 1));
        errors(i,cc,iter) = err;
        cc = cc + 1;
        for ii = 1:size(Y,2)
                nn = idx(ii);
                eta = eta0/(ii^alpha);
                y = Y(:,nn);
                x = C\y;
                C = (eta * y * x' + C) * (eye(k) - eta * (x * x')/(1 + eta * x' * x));
            if sum(err_iters == ii)
                beta = (C' * C)\C';
                X = beta * Y;
                err = mean(sum((Y - C * X).^2, 1));
                errors(i,cc,iter) = err;
                cc = cc + 1;
            end
        end
        disp(['PCA error ' num2str(err) ' eta ' num2str(eta0)]);
    end
end

%%
Y = data_pca';
Y = Y(:, randperm(size(Y,2)));
C = 0.001 * randn(size(Y,1),k);

eta0 = 10.0;
% eta = eta0;
alpha = 0.9;
idx = randperm(size(Y,2));
tic
for ii = 1:size(Y,2)
    nn = idx(ii);
%     eta = eta0/(floor(nn/1000)+1)^alpha;
%     eta = min(eta * 1.0005, eta0 * 1e8);
    eta = eta0/(ii^alpha);
%     beta = (C' * C)\C';
%     mu = beta * Y(:,nn);
    y = Y(:,nn);
    C = C + eta * y * (C' * y)';
%     m = ((ii-1) * m + Y(:,nn))/ii;
%     if ii == 1 || mod(ii,1000) == 0
%         beta = (C' * C)\C';
%         Ym = bsxfun(@minus, Y, m);
%         X = beta * Ym;
%         err = mean(sum((Ym - C * X).^2, 1));
%         disp(['iteration ' num2str(ii) ', PCA error ' num2str(err) ', mean error ' num2str(sqrt(sum((m_data - m').^2)))]);
%     end
    [C, ~] = qr(C,0);
end
toc
X = C\Y;
err = mean(sum((Y - C * X).^2, 1));
disp(['Oja error ' num2str(err)]);