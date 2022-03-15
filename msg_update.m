function [U, sig] = msg_update(U, sig, y, eta, k)
x = sqrt(eta) * U' * y;
x_orth = sqrt(eta) * y - U * x;
r = norm(x_orth);
if r > 0
    [V, sig] = eig([diag(sig) + x * x', r * x; r * x', r^2]);
    U = [U, x_orth/r] * V;
else
    [V, sig] = eig(diag(sig) + x * x');
    U = U * V;
end
sig = diag(sig);
S = project_gd_fzero(sig, k);
sig = max(0, min(sig + S,1));