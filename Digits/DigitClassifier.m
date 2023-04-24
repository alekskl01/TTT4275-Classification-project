%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

num_chuncks = 60;
chunck_size = 60000/num_chuncks;

for k = 1:num_test
    targets(testlab(k), k) = 1;
    test_sample = testv(k,:);
    closest_from_chuncks = zeros(num_chuncks, 2);
    for i = 1:num_chuncks
        trainv_chunck = trainv((chunck_size*(i-1)+1):chunck_size*i,:);
        distances =  dist(trainv_chunck, test_sample);
        disp(size(distances))
<<<<<<< HEAD
        [Min_dist,chunck_index] = min(distances,[],1);
        global_index = chunck_size*i + local_index;
        closest_from_chuncks(i,:) = [Min_dist; global_index];
=======
        [~,Shortest_index] = min(distances,[],1);
        Global_index = ...;
        Estimated_guess_i = trainlab(Global_index);
        outputs(Estimated_guess_i + 1, i) = 1;
>>>>>>> 8af423b67ce931a4c5b7a7156fe5f5d5f536e8d2
    end
    [~,I] = min(closest_from_chuncks(:,1),[],2);
    closest_distance_index = closest_from_chuncks(I,2);
    outputs(trainlab(closest_distance_index), k) = 1;
end


