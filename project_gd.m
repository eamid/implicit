function S = project_gd(sig, k)

d = length(sig);
[kappa, sig] = hist(sig, unique(sig));
[sig, idx] = sort(sig);
kappa = kappa(idx);
n = length(sig);
i = 1; j = 1; si = 0; sj = 0;
while i <= n
    if (i < j)
        S = (k - (si - sj) - (d - cj))/(cj - ci);
        if ((sig(i) + S) >= 0) && (sig(j-1) + S <= 1) && ((i <= 1) || ((sig(i-1) + S) <= 0))...
                && ((j >= n) || (sig(j+1) >= 1))
            break
        end
        if (j <= n) && ((sig(j) - sig(i)) <= 1)
            sj = sj + kappa(j) * sig(j);
            cj = cj + kappa(j);
            j = j + 1;
        else
            si = si + kappa(i) * sig(i);
            ci = ci + kappa(i);
            i = i + 1;
        end
    end
end
    