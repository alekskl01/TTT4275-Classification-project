%% Iris - Task 1
% By Sigurd von Brandis and Aleksander Klund
close all;

%% Load the data
x1all = load('class_1');
x2all = load('class_2');
x3all = load('class_3');
% 

%% Histograms;
% figure(1)
% histogram(x1all(:,1), 10);
% hold on;
% histogram(x2all(:,1), 10);
% hold on;
% histogram(x3all(:,1), 10);
% legend('class 1', 'class 2', 'class 3');
% sgtitle('Feature 1 for all classes')
% 
% figure(2);
% histogram(x1all(:,2), 10);
% hold on;
% histogram(x2all(:,2), 10);
% hold on;
% histogram(x3all(:,2), 10);
% legend('class 1', 'class 2', 'class 3');
% sgtitle('Feature 2 for all classes')  
% 
% figure(3);
% histogram(x1all(:,3), 10);
% hold on;
% histogram(x2all(:,3), 10);
% hold on;
% histogram(x3all(:,3), 10);
% legend('class 1', 'class 2', 'class 3');
% sgtitle('Feature 3 for all classes') 
% 
% figure(4);
% histogram(x1all(:,4), 10);
% hold on;
% histogram(x2all(:,4), 10);
% hold on;
% histogram(x3all(:,4), 10);
% legend('class 1', 'class 2', 'class 3');
% sgtitle('Feature 4 for all classes') 

%Parameters that classes are based upon.
% class_Setosa= x1all;
% class_Versicolor= x2all;
% class_Virginica= x3all;

class_Setosa= [x1all(:,1), x1all(:,3), x1all(:,4)];
class_Versicolor= [x2all(:,1), x2all(:,3), x2all(:,4)];
class_Virginica= [x3all(:,1), x3all(:,3), x3all(:,4)];

% class_Setosa= [x1all(:,3), x1all(:,4)];
% class_Versicolor= [x2all(:,3), x2all(:,4)];
% class_Virginica= [x3all(:,3), x3all(:,4)];

% class_Setosa= [x1all(:,4)];
% class_Versicolor= [x2all(:,4)];
% class_Virginica= [x3all(:,4)];

[Ntot,dimx] = size(class_Setosa);

%% Make Training and test sets
% TrainingSetLength = 21:50;
% TestSetLength = 1:20;
TrainingSetLength = 1:30;
TestSetLength = 31:50;
N_Training = length(TrainingSetLength);
N_Testing = length(TestSetLength);

%% Make Training Data
Tot_Training_Data = [class_Setosa(TrainingSetLength,:);
                     class_Versicolor(TrainingSetLength,:);
                     class_Virginica(TrainingSetLength,:)];

%% Make Test Data
Tot_Testing_Data = [class_Setosa(TestSetLength,:);
                    class_Versicolor(TestSetLength,:);
                    class_Virginica(TestSetLength,:)];

%% Make matrices used in confusion matrix
Correct_Answer_Training = [kron(ones(1,N_Training),[1; 0; 0]), kron(ones(1,N_Training),[0; 1; 0]), kron(ones(1,N_Training),[0; 0; 1])];

Correct_Answer_Testing = [kron(ones(1,N_Testing),[1; 0; 0]), kron(ones(1,N_Testing),[0; 1; 0]), kron(ones(1,N_Testing),[0; 0; 1])];

Measured_Answer_Training = zeros(size(Correct_Answer_Training));
Measured_Answer_Testing = zeros(size(Correct_Answer_Testing));

%% Train linear classifier
W = eye(3, dimx+1);
Alpha = 0.005;
iterations = 0;

while iterations < 30000
    gradientMSE = 0;
    for k = 1:3*N_Training
         xk = [Tot_Training_Data(k,:)'; 1];
         z = W * xk;
         gk = sigmoidFunction(z);
         tk = Correct_Answer_Training(:,k);
         gradientMSE = gradientMSE + ((gk-tk).*gk.*(1-gk))*xk';
    end
    
    W = W - Alpha*gradientMSE;

    iterations = iterations + 1;
end

%% Testing linear Classifier
for i = 1:length(Tot_Training_Data)
    x = [Tot_Training_Data(i,:)';1];
    z = W * x;
    g = sigmoidFunction(z);
    [val, class] = max(g);
    Measured_Answer_Training(class, i) = 1;
end

for i = 1:length(Tot_Testing_Data)
    x = [Tot_Testing_Data(i,:)';1];
    z = W * x;
    g = sigmoidFunction(z);
    [val, class] = max(g);
    Measured_Answer_Testing(class, i) = 1;
end

%% Prints and Comparisons
figure(5);
plotconfusion(Correct_Answer_Testing, Measured_Answer_Testing, 'Test set');
xticklabels({'Setosa', 'Versicolor', 'Virginica'})
yticklabels({'Setosa', 'Versicolor', 'Virginica'})

figure(6);
plotconfusion(Correct_Answer_Training, Measured_Answer_Training, 'Training set');
xticklabels({'Setosa', 'Versicolor', 'Virginica'})
yticklabels({'Setosa', 'Versicolor', 'Virginica'})

%% Functions

%Sigmoid function; 
function y = sigmoidFunction(z)
    % Compute the sigmoid function
    y = 1./(1 + exp(-z));
end