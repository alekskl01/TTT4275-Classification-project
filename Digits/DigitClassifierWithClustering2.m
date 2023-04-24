%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

num_classes = 10;
M = 64;

[classes, ~, idx_train] = unique(trainlab);
trainv_sorted = splitapply(@(x){x}, trainv, idx_train);

% TODO: fix this
for i = 1:num_classes
    [idx, C] = kmeans(trainv_sorted,M);
end

disp(C)

%Classifying
tic;
for k = 1:num_test
    targets(testlab(k)+1, k) = 1;
    test_sample = testv(k,:);
    closest_from_chuncks = zeros(num_chuncks, 2);
    distances =  dist(trainv, test_sample');
    [~, closest_distance_index] = min(distances,[],1);
    outputs(trainlab(closest_distance_index)+1, k) = 1;

    if mod(k, 500) == 0
        disp(k*100/num_test + "% done")
    end
end
toc

save('saveOutputsTask2.mat', "outputs")
save('saveTargets.mat', "targets")

%Confusion matrix and error rate

figure(1)
plotconfusion(targets, outputs, 'Classification result');