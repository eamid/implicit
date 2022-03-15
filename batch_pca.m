load coil20
k = 10;

data = X;
data = data/max(data(:));
data = bsxfun(@minus, data, mean(data,1));
tic
[coeff, S] = pca(data, 'NumComponents', k);
toc
err_pca = mean(sum((data - S * coeff').^2, 2));
disp(['PCA error ' num2str(err_pca)]);

Y = data';
C = 0.1 * randn(size(Y,1),k);

for ii = 1:50
    X = (C' * C)\C' * Y;
    C = Y * X'/(X * X');
    if ii == 1 || mod(ii,10) == 0
        err = mean(sum((Y - C * X).^2, 1));
        disp(['iteration ' num2str(ii) ', error ' num2str(err)]);
    end
end

    