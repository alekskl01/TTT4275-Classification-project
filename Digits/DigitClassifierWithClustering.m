%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

num_classes = 10;
M = 64;

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

%% Clustering

[classes, ~, idx_train] = unique(trainlab);
trainv_sorted = splitapply(@(x){x}, trainv, idx_train);

trainlab_cluster = zeros(M*num_classes, 1);
trainv_cluster = zeros(M*num_classes,784);

for i = 1:num_classes
    % Save class of all clusters
    trainlab_cluster(M*(i-1)+1:M*i, 1) = (i-1)*ones(M,1);

    [~, C_i] = kmeans(trainv_sorted{i,1},M);
    trainv_cluster(M*(i-1)+1:M*i,:) = C_i;
end


% save('Data/saveTrainvCluster.mat', 'trainv_cluster')
% save('Data/savetrainlab_cluster.mat', "trainlab_cluster")

%% Classifying
tic;
for k = 1:num_test
    targets(testlab(k)+1, k) = 1;
    test_sample = testv(k,:);
    
    %calculate distance, find index of closest centroid and check its class
    distances =  dist(trainv_cluster, test_sample');
    [~, closest_distance_index] = min(distances,[],1);
    outputs(trainlab_cluster(closest_distance_index)+1, k) = 1;

    if mod(k, 500) == 0
        disp(k*100/num_test + "% done")
    end
end
toc

% save('Data/saveOutputsTask2.mat', "outputs")
% save('Data/saveTargets.mat', "targets")

%Confusion matrix and error rate

figure(1)
plotconfusion(targets, outputs, 'Classification result');
xticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
yticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})