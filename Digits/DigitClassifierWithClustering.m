%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

% Perform k-means clustering on the training data
M = 10; % specify the number of clusters for k-means
[idxi, Ci] = kmeans(trainv, M);

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

% Classifying
tic;
for k = 1:num_test
    targets(testlab(k)+1, k) = 1;
    test_sample = testv(k,:);
    
    % Find the closest centroid from the k-means clustering
    closest_distance_index = dsearchn(Ci, test_sample);
    
    % Find the corresponding indices in the original training data
    closest_indices = find(idxi == closest_distance_index);
    
    % Compute distances from the closest samples in the training data
    distances = dist(trainv(closest_indices,:), test_sample');
    
    % Find the index with the minimum distance
    [~, closest_distance_index] = min(distances,[],1);
    
    outputs(trainlab(closest_indices(closest_distance_index))+1, k) = 1;

    if mod(k, 500) == 0
        disp(k*100/num_test + "% done")
    end
end
toc

save('saveOutputs.mat', "outputs")
save('saveTargets.mat', "targets")

% Confusion matrix and error rate

figure(1)
plotconfusion(targets, outputs, 'Classification result');
