%% Load the data
class_Setosa = load('class_1');
class_Versicolor = load('class_2');
class_Virginica = load('class_3');

%% Make Training and test sets
TrainingSetLength = 1:30;
TestSetLength = 31:50;
N_Training = length(TrainingSetLength);
N_Test = length(TestSetLength);

%% Make Training Data
Tot_Training_Data = [class_Setosa{TrainingSetLenght,:};
                     class_Versicolor{TrainingSetLength,:};
                     class_Virginica{TrainingSetLength,:}];

%% Make Test Data
Tot_Testing_Data = [class_Setosa{TestSetLength,:};
                    class_Versicolor{TestSetLength,:};
                    class_Versicolor{TestSetLength,:}];

%% Make matrices used in confusion matrix
Correct_Answer_Training = ;
Correct_Answer_Testing = ;
Measured_Answer_training = ;
Measured_Answer_Testing = ;


%% Train linear classifier

%% Testing linear Classifier

%% Prints and Comparisons