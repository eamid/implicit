close all
clear h
cols = brewermap(10, 'Paired');
cols = cols(2:2:end,:);
load centered_pca_sanger_mnist.mat
error_sanger = error;
load centered_pca_results_mnist.mat

N = 70000;

x = 0:5000:N;
error = error(:,:,:,6:10);
for j = 1:length(error(6,1,:,1))
    error(6,1,j,1) = error(6,1,j,2);
end

% for i = [1:8, 10]
%     for j = 1:15
%         error(6,2,j,i) = error(6,2,j,9);
%     end
% end

% error = error(:,:,:,6:end);

error_all = error;
error_all = mean(error_all, 4);
error_sanger_all = mean(error_sanger, 4);
runtime = mean(runtime, 4);
k_values = [5,10,20,50];

for kk = 1:length(k_values)
    k = k_values(kk);
    times = squeeze(runtime(:,kk,:));
    errors = squeeze(error_all(:,kk,:));
    pca_error = errors(1,1);
    errors = (errors/pca_error -1) * 100;
    errors_sanger = squeeze(error_sanger_all(:,kk,:));
    errors_sanger = (errors_sanger/pca_error - 1) * 100;
%     figure
%     hold on
%     ylim_up = max(errors(:,1));
%     h{1} = plot(x, mean(errors(2,:), 1), '-^', 'color', cols(1,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
%     h{2} = plot(x, mean(errors(3,:), 1), '-x', 'color', cols(2,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
%     h{3} = plot(x, mean(errors(6,:), 1), '-s', 'color', cols(3,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
%     h{4} = plot(x, mean(errors(4,:), 1), '-d', 'color', cols(4,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
% %     h{1} = plot(x, mean(errors(2,:), 1), '-', 'color', cols(1,:), 'linewidth', 3);
% %     h{2} = plot(x, mean(errors(3,:), 1), '-', 'color', cols(2,:), 'linewidth', 3);
% %     h{3} = plot(x, mean(errors(6,:), 1), '-', 'color', cols(3,:), 'linewidth', 3);
% %     h{4} = plot(x, mean(errors(4,:), 1), '-', 'color', cols(4,:), 'linewidth', 3);
%     ylim([0, ylim_up + 5])
%     xlim([0, max(x)])
%     set(gca, 'YScale', 'log')
%     if kk == 1
%         legend([h{:}], {'Oja', 'Krasulina', 'Incremental', 'Imp. Krasulina'}, 'Interpreter','latex', 'fontsize', 30)
%     end
%     xlabel('number of iterations', 'Interpreter','latex', 'fontsize', 30)
%     ylabel('excess loss', 'Interpreter','latex', 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     set(gca,'TickLabelInterpreter','latex')
%     title(['Distributed Update - CIFAR-10 ($k=' num2str(k) '$)'], 'Interpreter','latex', 'fontsize', 32)
%     set(gcf, 'position', [100 100 700 600]);
%     
    figure
    hold on
    ylim_up = max(errors(:,1));
%     h{1} = plot(times(2,:), errors(2,:), '-', 'color', cols(1,:), 'linewidth', 3);
%     h{2} = plot(times(3,:), errors(3,:), '-', 'color', cols(2,:), 'linewidth', 3);
%     h{3} = plot(times(6,:), errors(6,:), '-', 'color', cols(3,:), 'linewidth', 3);
%     h{4} = plot(times(4,:), errors(4,:), '-', 'color', cols(4,:), 'linewidth', 3);
    h{1} = plot(times(2,:), errors(2,:), '-^', 'color', cols(1,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
    h{2} = plot(times(3,:), errors(3,:), '-x', 'color', cols(2,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
    h{3} = plot(times(6,:), errors(6,:), '-s', 'color', cols(3,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
    h{4} = plot(times(4,:), errors_sanger, '-o', 'color', cols(5,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
    h{5} = plot(times(4,:), errors(4,:), '-d', 'color', cols(4,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1,1,1]);
    ylim([0, ylim_up + 5])
    xlim([0, max(times([2,3,6,4],end)) + 5])
    set(gca, 'YScale', 'log')
    if kk == 1
        legend([h{:}], {'Oja', 'Krasulina', 'Incremental', 'Sanger', 'Imp. Krasulina'}, 'Interpreter','latex', 'fontsize', 30)
    end
    xlabel('runtime', 'Interpreter','latex', 'fontsize', 30)
    ylabel('percent excess loss', 'Interpreter','latex', 'fontsize', 30)
    set(gca, 'fontsize', 25)
    set(gca,'TickLabelInterpreter','latex')
    title(['Centered PCA$\!$ -$\!$ MNIST ($k\!=\!' num2str(k) '$)'], 'Interpreter','latex', 'fontsize', 30)
    set(gcf, 'position', [100 100 600 500]);
end