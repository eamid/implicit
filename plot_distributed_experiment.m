clear h
cols = brewermap(10, 'Paired');
cols = cols(2:2:end,:);
load distributed_pca_results_mnist_k_5.mat
load distributed_pca_sanger_mnist_k_5.mat
% load distributed_pca_implicit_krasulina_cifar10_k_20.mat
error_imp_krasulina = error_imp_krasulina(:,:,1:5);

N = 7000;

ylim_up = 0;
figure
hold on
% h{1} = plot([0 7000], [1 1], 'k--', 'linewidth', 3);

x = 0:1:7;
error_oja = (error_oja/error_pca - 1) * 100;
error_oja_m = mean(error_oja(1:10,1:end,:),3);
error_oja_c = mean(error_oja(11,1:end,:),3);
h{1} = plot(x, mean(error_oja_m, 1), '--', 'color', cols(1,:), 'linewidth', 3);
h{2} = plot(x, error_oja_c, '-^', 'color', cols(1,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);

ylim_up = max(ylim_up, mean(error_oja_m(:,1)));
ylim_up = max(ylim_up, error_oja_c(1));
% astd = std(error_oja_m,1);
% mu = mean(error_oja_m,1);
% fill([x fliplr(x)],[mu+astd fliplr(mu-astd)],'b', 'FaceAlpha', 0.6,'linestyle','none','HandleVisibility','off');

error_krasulina = (error_krasulina/error_pca - 1) * 100;
error_krasulina_m = mean(error_krasulina(1:10,1:end,:),3);
error_krasulina_c = mean(error_krasulina(11,1:end,:),3);
h{3} = plot(x, mean(error_krasulina_m, 1), '--', 'color', cols(2,:), 'linewidth', 3);
h{4} = plot(x, error_krasulina_c, '-x', 'color', cols(2,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);

ylim_up = max(ylim_up, mean(error_krasulina_m(:,1)));
ylim_up = max(ylim_up, error_krasulina_c(1));

error_incremental = (error_incremental/error_pca - 1) * 100;
error_incremental_m = mean(error_incremental(1:10,1:end,:),3);
error_incremental_c = mean(error_incremental(11,1:end,:),3);
h{5} = plot(x, mean(error_incremental_m, 1), '--', 'color', cols(3,:), 'linewidth', 3);
h{6} = plot(x, error_incremental_c, '-s', 'color', cols(3,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
ylim_up = max(ylim_up, mean(error_incremental_m(:,1)));
ylim_up = max(ylim_up, error_incremental_c(1));

error_sanger = (error_sanger/error_pca - 1) * 100;
error_sanger_m = mean(error_sanger(1:10,1:end,:),3);
error_sanger_c = mean(error_sanger(11,1:end,:),3);
h{7} = plot(x, mean(error_sanger_m, 1), '--', 'color', cols(5,:), 'linewidth', 3);
h{8} = plot(x, error_sanger_c, '-o', 'color', cols(5,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
ylim_up = max(ylim_up, mean(error_sanger_m(:,1)));
ylim_up = max(ylim_up, error_sanger_c(1));

error_imp_krasulina = (error_imp_krasulina/error_pca - 1) * 100;
error_imp_krasulina_m = mean(error_imp_krasulina(1:10,1:end,:),3);
error_imp_krasulina_c = mean(error_imp_krasulina(11,1:end,:),3);
h{9} = plot(x, mean(error_imp_krasulina_m, 1), '--', 'color', cols(4,:), 'linewidth', 3);
h{10} = plot(x, error_imp_krasulina_c, '-d', 'color', cols(4,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
ylim_up = max(ylim_up, mean(error_imp_krasulina_m(:,1)));
ylim_up = max(ylim_up, error_imp_krasulina_c(1));

for i = 1:2:7
    set(get(get(h{i},'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
ylim([0, ylim_up + 5])
xlim([0, max(x) + 0.12])
set(gca, 'YScale', 'log')
legend([h{2:2:end}], {'Oja', 'Krasulina', 'Incremental', 'Sanger', 'Imp. Krasulina'}, 'Interpreter','latex', 'fontsize', 30)
xlabel('sync step', 'Interpreter','latex', 'fontsize', 30)
ylabel('percent excess loss', 'Interpreter','latex', 'fontsize', 30)
set(gca, 'fontsize', 25)
set(gca,'TickLabelInterpreter','latex')
title('Distributed Update - MNIST ($k=5$)', 'Interpreter','latex', 'fontsize', 30)
% title('Distributed Update - MNIST ($k=20$)', 'Interpreter','latex', 'fontsize', 30)

set(gcf, 'position', [100 100 700 600]);