clc
load mnist
k = 5;

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
Y = data';
C = 0.001 * randn(size(Y,1),k);
m = 0.01 * randn(size(Y,1),1);

eta0 = 100.0;
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
    x = C\(y - m);
    m = (eta * (y - C * x) + m)/(eta + 1);
    C = (eta * (y - m) * x' + C) * (eye(k) - eta * (x * x')/(1 + eta * x' * x));
%     m = ((ii-1) * m + Y(:,nn))/ii;
%     if ii == 1 || mod(ii,1000) == 0
%         beta = (C' * C)\C';
%         Ym = bsxfun(@minus, Y, m);
%         X = beta * Ym;
%         err = mean(sum((Ym - C * X).^2, 1));
%         disp(['iteration ' num2str(ii) ', PCA error ' num2str(err) ', mean error ' num2str(sqrt(sum((m_data - m').^2)))]);
%     end
end
toc
beta = (C' * C)\C';
Ym = bsxfun(@minus, Y, m);
X = beta * Ym;
err = mean(sum((Ym - C * X).^2, 1));
disp(['PCA error ' num2str(err) ', mean error ' num2str(mean((m_data - m').^2))]);