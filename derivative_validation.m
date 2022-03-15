pert = 1e-11 * randn(d,k);
C = 2 * randn(d,k);
y = randn(d,1);
f = @(C)(y - C * pinv(C) * y)' * (y - C * pinv(C) * y);

df = zeros(d,k);
for dd = 1:d
    for kk = 1:k
        mask = zeros(d,k);
        mask(dd,kk) = 1;
        df(dd,kk) = (f(C + mask .* pert) - f(C))/pert(dd,kk);
    end
end

g = @(C) 2 * ((C * pinv(C) - eye(size(C,1))) * y * y' * pinv(C)' - pinv(C)' * C' * (C * pinv(C) - eye(size(C,1))) * y * y' * pinv(C)');
g = @(C) -2 * (

disp([df, nan(d,1), g(C)])