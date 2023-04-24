%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

num_chuncks = 60;
chunck_size = 60000/num_chuncks;

%Classifying
for k = 1:num_test
    targets(testlab(k)+1, k) = 1;
    test_sample = testv(k,:);
    closest_from_chuncks = zeros(num_chuncks, 2);
    for i = 1:num_chuncks
        trainv_chunck = trainv((chunck_size*(i-1)+1):chunck_size*i,:);
        distances =  dist(trainv_chunck, test_sample');
        [Min_dist, local_index] = min(distances,[],1);
        global_index = chunck_size*(i-1) + local_index;
        closest_from_chuncks(i,:) = [Min_dist global_index];
    end
    [~,I] = min(closest_from_chuncks(:,1),[],1);
    closest_distance_index = closest_from_chuncks(I,2);
    outputs(trainlab(closest_distance_index)+1, k) = 1;
end

%Confusion matrix and error rate

figure(1)
plotconfusion(targets, outputs, 'Classification result');

