%% Iris - Task 1
% By Sigurd von Brandis and Aleksander KLund

%% Load the data
class_Setosa = load('class_1');
class_Versicolor = load('class_2');
class_Virginica = load('class_3');

%% Make Training and test sets
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
                    class_Versicolor(TestSetLength,:)];

%% Make matrices used in confusion matrix
Correct_Answer_Training = [kron(ones(1,N_Training),[1; 0; 0]);
                           kron(ones(1,N_Training),[0; 1; 0]); 
                           kron(ones(1,N_Training),[0; 0; 1])];

Correct_Answer_Testing = [kron(ones(1,N_Testing),[1; 0; 0]);
                          kron(ones(1,N_Testing),[0; 1; 0]); 
                          kron(ones(1,N_Testing),[0; 0; 1])];

Measured_Answer_Taining = zeros(size(Correct_Answer_Training));
Measured_Answer_Testing = zeros(size(Correct_Answer_Testing));

%% Train linear classifier
W = eye(3, 5);
Alpha = 0.01;
while true
    
    gradientMSE = 0;

    for k = 1:3*N_Training
         xk = [Tot_Training_Data(k,:)'; 1];
         gk = sigmoid(W*xk);
         tk = Correct_Answer_Training;
         gradientMSE = gradientMSE + 0.5*(gk-tk).'*(gk-tk);
    end
    
    W = W - Alpha*gradientMSE;
    
    if abs(gradientMSE) < 0.2
        break
    end
end

%% Testing linear Classifier

%% Prints and Comparisons