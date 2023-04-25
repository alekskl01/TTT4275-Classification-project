%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

num_classes = 10;
M = 64;
K =7;

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

%% Classifying
tic;
for i = 1:num_test
    targets(testlab(i)+1, i) = 1;
    test_sample = testv(i,:);
    distances =  dist(trainv_cluster, test_sample');
    [~, indexes_sorted] = sort(distances);
    K_closest_indexes = indexes_sorted(1:K);
    K_closest_classes = trainlab_cluster(K_closest_indexes);
    Most_frequent_class = mode(K_closest_classes);
    outputs((Most_frequent_class+1), i) = 1;

    if mod(i, 500) == 0
        disp(i*100/num_test + "% done")
    end
end
toc

% save('Data/saveOutputsTaskKNN.mat', "outputs")
% save('Data/saveTargets.mat', "targets")

%Confusion matrix and error rate
figure(1)
plotconfusion(targets, outputs, 'Classification result');
xticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
yticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})