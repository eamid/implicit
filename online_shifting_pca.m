clc
load cifar10
k = 5;

data = X;
data = data/max(data(:));
data = bsxfun(@minus, data, mean(data,1));
tic
[coeff, S] = pca(data, 'NumComponents', k);
toc
err_pca = mean(sum((data - S * coeff').^2, 2));
disp(['PCA error ' num2str(err_pca)]);
%%
Y = data';
C = 0.1 * randn(size(Y,1),k);
errors = zeros(1,121);
eta0 = 1.0;
% eta = eta0;
alpha = 0.7;
[~,idx] = sort(L);
tic
cc = 1;
for ii = 1:size(Y,2)
    nn = idx(ii);
%     eta = eta0/(floor(nn/1000)+1)^alpha;
%     eta = min(eta * 1.0005, eta0 * 1e8);
    eta = eta0/(ii^alpha);
%     beta = (C' * C)\C';
%     mu = beta * Y(:,nn);
    mu = C\Y(:,nn);
    C = (eta * Y(:,nn) * mu' + C) * (eye(k) - eta * (mu * mu')/(1 + eta * mu' * mu));
    if ii == 1 || mod(ii,500) == 0
        beta = (C' * C)\C';
        X = beta * Y;
        err = mean(sum((Y - C * X).^2, 1));
        errors(cc) = err;
        cc = cc + 1;
        disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
    end
end
toc
beta = (C' * C)\C';
X = beta * Y;
err = mean(sum((Y - C * X).^2, 1));
disp(['final error ' num2str(err)]);

%% Oja's
Y = data';
C = randn(size(Y,1),k);
[C,~] = qr(C,0);
errors_oja = zeros(1,121);

eta0 = 0.1;
% eta = eta0;
alpha = 0.7;
[~,idx] = sort(L);
cc = 1;
for ii = 1:size(Y,2)
    nn = idx(ii);
    eta = eta0/(ii^alpha);
    C = C + eta * Y(:,nn) * Y(:,nn)' * C;
    [C,~] = qr(C,0);
    if ii == 1 || mod(ii,500) == 0
        X = C' * Y;
        err = mean(sum((Y - C * X).^2, 1));
        errors_oja(cc) = err;
        cc = cc + 1;
        disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
    end
end
X = C' * Y;
err = mean(sum((Y - C * X).^2, 1));
disp(['final error ' num2str(err)]);