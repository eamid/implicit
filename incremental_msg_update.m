function U = incremental_msg_update(U, y, k)
x = U' * y;
x_orth = y - U * x;
r = norm(x_orth);
if r > 0
    [V,sig] = eig([eye(k) + x * x', r * x; r * x', r^2]);
    sig = diag(sig);
    idx = sig > min(sig);
    U = [U, x_orth/r] * V(:,idx);
else
    [V,~] = eig(eye(k) + x * x');
    U = U * V;
end