%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

%Classifying
tic;
for k = 1:num_test
    targets(testlab(k)+1, k) = 1;
    test_sample = testv(k,:);
    distances =  dist(trainv, test_sample');
    [~, closest_distance_index] = min(distances,[],1);
    outputs(trainlab(closest_distance_index)+1, k) = 1;

    if mod(k, 500) == 0
        disp(k*100/num_test + "% done")
    end
end
toc

save('saveOutputsTask1.mat', "outputs")
save('saveTargets.mat', "targets")

%Confusion matrix and error rate

figure(1)
plotconfusion(targets, outputs, 'Classification result');