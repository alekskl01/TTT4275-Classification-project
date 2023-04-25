%% Classification - Digits 
% By Sigurd von Brandis and Aleksander Klund

outputs = zeros(10, num_test);
targets = zeros(10, num_test);

%% Classifying
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

% save('Data/saveOutputsTask1.mat', "outputs")
% save('Data/saveTargets.mat', "targets")

%% Plot

num_pictures = 16;

incorrect_plot = zeros(num_pictures, 2); 
correct_plot = zeros(num_pictures, 2);

a = 0;
b = 0;
for k = 1:num_test
    output_value = find(outputs(:,k)) - 1;
    if output_value ~= testlab(k) && (a < num_pictures)
        a = a + 1;
        incorrect_plot(a,:) = [k; output_value];
    elseif output_value == testlab(k) && (b < num_pictures)
        b = b + 1;
        correct_plot(b,:) = [k; output_value];
    end
end

% Incorrectly Classified
figure(1);
sgtitle("Incorrectly classified images")
for n = 1:num_pictures
    x = zeros(28,28);
    x(:) = testv(incorrect_plot(n,1),:);
    x = fliplr(x);
    x = rot90(x);
    subplot(4,4,n);
    image(x);
    Correct = testlab(incorrect_plot(n,1));
    Estimated = incorrect_plot(n,2);
    title("Correct: " + Correct +", " + "Esimated: " + Estimated)
end

% Correctly Classified
figure(2);
sgtitle("Correctly classified images")
for n = 1:num_pictures
    x = zeros(28,28);
    x(:) = testv(correct_plot(n,1),:);
    x = fliplr(x);
    x = rot90(x);
    subplot(4,4,n);
    image(x);
    Correct = testlab(correct_plot(n,1));
    Estimated = correct_plot(n,2);
    title("Correct: " + Correct +", " + "Esimated: " + Estimated)
end


figure(3)
plotconfusion(targets, outputs, 'Classification result');
xticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
yticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})