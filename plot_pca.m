cols = brewermap(6, 'Set1');
close all
figure
hold on
msize = 15;
lw = 3;
lr = 1;
h{1} = semilogy(k_values, mean(error(1,:,lr,:),4), '-^', 'Color', cols(1,:), 'linewidth', lw, 'markersize', msize);
h{2} = semilogy(k_values, mean(error(2,:,lr,:),4), '-s', 'Color', cols(2,:), 'linewidth', lw, 'markersize', msize);
h{3} = semilogy(k_values, mean(error(3,:,lr,:),4), '-d', 'Color', cols(3,:), 'linewidth', lw, 'markersize', msize);
h{4} = semilogy(k_values, mean(error(4,:,lr,:),4), '-v', 'Color', cols(4,:), 'linewidth', lw, 'markersize', msize);
h{5} = semilogy(k_values, mean(error(5,:,lr,:),4), '-o', 'Color', cols(5,:), 'linewidth', lw, 'markersize', msize);
h{6} = semilogy(k_values, mean(error(6,:,lr,:),4), '-x', 'Color', cols(6,:), 'linewidth', lw, 'markersize', msize);


legend([h{:}], {'Batch PCA', 'Oja', 'Krasulina', 'Imp. Oja', 'Imp. Krasulina', 'Incremental'}, 'Interpreter','latex', 'fontsize', 24)
xlabel('$k$', 'Interpreter','latex')
ylabel('Compression Error', 'Interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca, 'fontSize', 35)
set(gcf, 'position', [100 100 700 600]);
title('Scene (centered)', 'Interpreter','latex', 'fontSize', 40)
tightfig

%%
cols = brewermap(5, 'Set1');
close all
figure
hold on
msize = 15;
lw = 3;
lr = 1;
h{1} = semilogx(k_values, mean(runtime(1,:,lr,:),4), '-^', 'Color', cols(1,:), 'linewidth', lw, 'markersize', msize);
h{2} = semilogx(k_values, mean(runtime(2,:,lr,:),4), '-s', 'Color', cols(2,:), 'linewidth', lw, 'markersize', msize);
h{3} = semilogx(k_values, mean(runtime(3,:,lr,:),4), '-d', 'Color', cols(3,:), 'linewidth', lw, 'markersize', msize);
h{4} = semilogx(k_values, mean(runtime(4,:,lr,:),4), '-v', 'Color', cols(4,:), 'linewidth', lw, 'markersize', msize);
h{5} = semilogx(k_values, mean(runtime_krasulina,2), '-o', 'Color', cols(5,:), 'linewidth', lw, 'markersize', msize);


% legend([h{:}], {'Batch PCA', 'Oja', 'Krasulina', 'Imp. Oja', 'Imp. Krasulina'}, 'Interpreter','latex', 'fontsize', 24)
xlabel('$k$', 'Interpreter','latex')
ylabel('Runtime', 'Interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca, 'fontSize', 35)
set(gcf, 'position', [100 100 700 600]);
title('USPS', 'Interpreter','latex', 'fontSize', 40)
tightfig

