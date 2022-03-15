clear h
clc
cols = brewermap(8, 'Paired');
cols = cols(2:2:end,:);
load eta_sensitivity_results_mnist_k_5.mat

ylim_up = 0;
figure
hold on

% error_all = error_all(:,:,:,1:3);
error_all = mean(error_all, 4);
error_all = (error_all/error_pca - 1) * 100;

x = 0:5000:70000;

error_oja = squeeze(error_all(1,:,:));
error_oja_m = error_oja(2,:);
[~,idx] = max(error_oja([1,3],end));
idx = idx * 2 - 1;
error_oja_c = error_oja(idx,:);
h{1} = plot(x, error_oja_c, '--', 'color', cols(1,:), 'linewidth', 3);
h{2} = plot(x, error_oja_m, '-', 'color', cols(1,:), 'linewidth', 3);


ylim_up = max(ylim_up, mean(error_oja_m(1)));
ylim_up = max(ylim_up, error_oja_c(1));
% astd = std(error_oja_m,1);
% mu = mean(error_oja_m,1);
% fill([x fliplr(x)],[mu+astd fliplr(mu-astd)],'b', 'FaceAlpha', 0.6,'linestyle','none','HandleVisibility','off');

error_krasulina = squeeze(error_all(2,:,:));
error_krasulina_m = error_krasulina(2,:);
[~,idx] = max(error_krasulina([1,3],end));
idx = idx * 2 - 1;
error_krasulina_c = error_krasulina(idx,:);
h{3} = plot(x, error_krasulina_c, '--', 'color', cols(2,:), 'linewidth', 3);
h{4} = plot(x, error_krasulina_m, '-', 'color', cols(2,:), 'linewidth', 3);

ylim_up = max(ylim_up, mean(error_krasulina_m(1)));
ylim_up = max(ylim_up, error_krasulina_c(1));

error_incremental = squeeze(error_all(4,:,:));
error_incremental_m = error_incremental(2,:);
error_incremental_c = error_incremental(1,:);
h{5} = plot(x, error_incremental_c, '-', 'color', cols(3,:), 'linewidth', 3);
h{6} = plot(x, error_incremental_m, '-', 'color', cols(3,:), 'linewidth', 3);

ylim_up = max(ylim_up, mean(error_incremental_m(1)));
ylim_up = max(ylim_up, error_incremental_c(1));

error_imp_krasulina = squeeze(error_all(3,:,:));
error_imp_krasulina_m = error_imp_krasulina(2,:);
[~,idx] = max(error_imp_krasulina([1,3],end));
idx = idx * 2 - 1;
error_imp_krasulina_c = error_imp_krasulina(idx,:);
h{7} = plot(x, error_imp_krasulina_c, '--', 'color', cols(4,:), 'linewidth', 3);
h{8} = plot(x, error_imp_krasulina_m, '-', 'color', cols(4,:), 'linewidth', 3);

ylim_up = max(ylim_up, mean(error_imp_krasulina_m(1)));
ylim_up = max(ylim_up, error_imp_krasulina_c(1));

for i = 1:2:7
    set(get(get(h{i},'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
ylim([0, ylim_up + 5])
xlim([0, max(x)])
set(gca, 'YScale', 'log')
legend([h{2:2:8}], {'Oja', 'Krasulina', 'Incremental', 'Imp. Krasulina'}, 'Interpreter','latex', 'fontsize', 30)
xlabel('number of iterations', 'Interpreter','latex', 'fontsize', 25)
ylabel('percent loss', 'Interpreter','latex', 'fontsize', 25)
set(gca, 'fontsize', 25)
set(gca,'TickLabelInterpreter','latex')
title('$\eta$-Sensitivity - MNIST ($k=5$)', 'Interpreter','latex', 'fontsize', 32)
set(gcf, 'position', [100 100 700 600]);