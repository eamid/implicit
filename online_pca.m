clc
load mnist
k = 10;

data = X;
[N, dim] = size(X);
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
Cinv = inv(C' * C);

eta0 = 0.1;
alpha = 0.5;
idx = randperm(size(Y,2));
tic
for ii = 1:size(Y,2)
    nn = idx(ii);
    y = Y(:,nn);
    eta = eta0/(ii^alpha);
%     beta = (C' * C)\C';
    mu = Cinv * (C' * y);
    yc = C'* y;
    eta = eta/(1 + eta * (mu' * mu));
%     mu = C\Y(:,nn);
%     denom = (1 + eta * sum(mu.^2));
%     C = (eta * Y(:,nn) * mu' + C) * (eye(k) - eta * (mu * mu')/(1 + eta * mu' * mu));
    C = C + eta * (y - C * mu) * mu';
    U = [mu, eta * yc];
    V = [eta * yc + eta^2 * (y' * y) * mu, mu];
    Cinv = Cinv - Cinv * U/(eye(2) + V' * Cinv * U) * (V' * Cinv); 
%     if ii == 1 || mod(ii,1000) == 0
%         beta = (C' * C)\C';
%         X = beta * Y;
%         err = mean(sum((Y - C * X).^2, 1));
%         disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
%     end
end
toc
beta = (C' * C)\C';
X = beta * Y;
err = mean(sum((Y - C * X).^2, 1));
disp(['final error ' num2str(err)]);

%% Oja's
Y = data';
C = randn(size(Y,1),k);
[C,Q] = qr(C,0);

eta0 = 0.1;
% eta = eta0;
alpha = 0.7;
idx = randperm(size(Y,2));
for ii = 1:size(Y,2)
    nn = idx(ii);
    eta = eta0/(ii^alpha);
    C = C + eta * Y(:,nn) * Y(:,nn)' * C;
    [C,~] = qr(C,0);
    if ii == 1 || mod(ii,1000) == 0
        X = C' * Y;
        err = mean(sum((Y - C * X).^2, 1));
        disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
    end
end
X = C' * Y;
err = mean(sum((Y - C * X).^2, 1));
disp(['final error ' num2str(err)]);
%% Dima
load centeredset G;
Y = zeros(300,165);
k = 10;
for ii = 1:165
    y = G(:,:,ii);
    Y(:,ii) = y(:);
end
[dim, N] = size(Y);
% Y = data';
W = eye(dim)/dim;
I = eye(dim);
eta0 = 0.1;
alpha = 0.7;
idx = randperm(size(Y,2));
for ii = 1:size(Y,2)
    nn = idx(ii);
    eta = eta0/(ii^alpha);
    y = Y(:,nn); y = y/norm(y);
    W = pcaupdate(W, y * y', eta, k);
    if ii == 1 || mod(ii,10) == 0
        P = maxlikepca(W,k);
        err = trace((I-P) * (Y * Y'))/N;
        disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
    end
end


%% Capped MSG
Y = data';
C = randn(size(Y,1));
C = C' * C;
[U,D] = eig(C);
U = U(:,1:(k+1));
sig = diag(D);
sig = sig(1:(k+1));
S = project_gd_fzero(sig, k);
sig = max(0, min(sig + S,1));

eta0 = 1.0;
alpha = 0.5;
idx = randperm(size(Y,2));
for ii = 1:size(Y,2)
    nn = idx(ii);
    eta = eta0/(ii^alpha);
    [U, sig] = msg_update(U,sig,Y(:,nn),eta,k);
    if ii == 1 || mod(ii,1000) == 0
        C = U * diag(sig) * U';
        err = trace((eye(size(Y,1)) - C) * (Y * Y'))/size(Y,2);
        disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
        disp(sig)
    end
end

%% Incremental MSG
Y = data';
C = randn(size(Y,1));
C = C' * C;
[U,~] = eig(C);
U = U(:,1:k);

idx = randperm(size(Y,2));
tic
for ii = 1:size(Y,2)
    nn = idx(ii);
    U = incremental_msg_update(U,Y(:,nn),k);
%     if ii == 1 || mod(ii,1000) == 0
%         C = U * U';
%         err = trace((eye(size(Y,1)) - C) * (Y * Y'))/size(Y,2);
%         disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
%     end
end
toc
