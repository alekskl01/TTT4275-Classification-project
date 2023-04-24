%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

num_classes = 10;
M = 64;

%% Clustering
outputs = zeros(10, num_test);
targets = zeros(10, num_test);

tic;
[classes, ~, idx_train] = unique(trainlab);
trainv_sorted = splitapply(@(x){x}, trainv, idx_train);

trainlab_cluster = zeros(M*num_classes, 1);

trainv_cluster = zeros(M*num_classes,784);
for i = 1:num_classes
    trainlab_cluster(M*(i-1)+1:M*i, 1) = (i-1)*ones(M,1);

    [~, C_i] = kmeans(trainv_sorted{i,1},M);
    trainv_cluster(M*(i-1)+1:M*i,:) = C_i;
end

%% Classifying
for k = 1:num_test
    targets(testlab(k)+1, k) = 1;
    test_sample = testv(k,:);
    distances =  dist(trainv_cluster, test_sample');
    [~, closest_distance_index] = min(distances,[],1);
    outputs(trainlab_cluster(closest_distance_index)+1, k) = 1;

    if mod(k, 500) == 0
        disp(k*100/num_test + "% done")
    end
end
toc

%save('saveOutputsTask2.mat', "outputs")
%save('saveTargets.mat', "targets")

%Confusion matrix and error rate

figure(1)
plotconfusion(targets, outputs, 'Classification result');