%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

num_chuncks = 60;
chunck_size = 60000/num_chuncks;

for k = 1:num_test
    correct_class = testlab(k);
    test_sample = testv(k,:);
    for i = 1:num_chuncks
        trainv_chunck = trainv((chunck_size*(i-1)+1):chunck_size*i,:);
        distances =  dist(trainv_chunck, test_sample);
        disp(size(distances))
        [~,Shortest_index] = min(distances,[],1);
        Global_index = ...;
        Estimated_guess_i = trainlab(Global_index);
        outputs(Estimated_guess_i + 1, i) = 1;
    end
end
