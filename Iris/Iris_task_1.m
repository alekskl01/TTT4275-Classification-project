%% Iris - Task 1
% By Sigurd von Brandis and Aleksander KLund

%% Load the data
x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');
% 
class_Setosa= [x1all(:,1) x1all(:,2) x1all(:,3) x1all(:,4)];
class_Versicolor= [x2all(:,1) x2all(:,2) x2all(:,3) x2all(:,4)];
class_Virginica= [x3all(:,1) x3all(:,2) x3all(:,3) x3all(:,4)];

% class_Setosa= [x1all(:,2)];
% class_Versicolor= [x2all(:,2)];
% class_Virginica= [x3all(:,2)];
% % 
% class_Setosa= [x1all(:,4)];
% class_Versicolor= [x2all(:,4)];
% class_Virginica= [x3all(:,4)];

[Ntot,dimx] = size(class_Setosa);

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
Correct_Answer_Training = [kron(ones(1,N_Training),[1; 0; 0]) kron(ones(1,N_Training),[0; 1; 0]) kron(ones(1,N_Training),[0; 0; 1])];

Correct_Answer_Testing = [kron(ones(1,N_Testing),[1; 0; 0]) kron(ones(1,N_Testing),[0; 1; 0]) kron(ones(1,N_Testing),[0; 0; 1])];

Measured_Answer_Training = zeros(size(Correct_Answer_Training));
Measured_Answer_Testing = zeros(size(Correct_Answer_Testing));

%% Train linear classifier
W = eye(3, dimx+1);
Alpha = 0.01;
gradientMSE = 0;

while true
    gradientMSE = 0;
    for k = 1:3*N_Training
         xk = [Tot_Training_Data(k,:)'; 1];
         z = W * xk;
         gk = sigmoidFunction(z);
         tk = Correct_Answer_Training(:,k);
         gradientMSE = gradientMSE + ((gk-tk).*gk.*(1-gk))*xk';
    end
    
    W = W - Alpha*gradientMSE;
    if abs(gradientMSE) < 0.01
        break
    end
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
figure(1);
plotconfusion(Correct_Answer_Testing, Measured_Answer_Testing, 'Test set');

figure(2);
plotconfusion(Correct_Answer_Training, Measured_Answer_Training, 'Training set');

%% Functions

%Sigmoid function; 
function y = sigmoidFunction(z)
    % Compute the sigmoid function
    y = 1./(1 + exp(-z));
end
