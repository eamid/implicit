function S = project_gd_fzero(sig, k)

f = @(S) (sum(max(0, min(sig + S, 1))) - k);
S = fzero(f,0);