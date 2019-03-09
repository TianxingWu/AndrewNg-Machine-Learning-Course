%% Initialization
clear ; close all; clc


%% =========== Train Linear SVM for Spam Classification ========
% Load the Spam Email dataset
% You will have X_train, y_train in your environment
load('trainSet.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.03;
model = svmTrain(X_train, y_train, C, @linearKernel);

p = svmPredict(model, X_train);

fprintf('Training Accuracy: %f\n', mean(double(p == y_train)) * 100);

%% =================== Validation Spam Classification ================
% Load the validation dataset
% You will have X_val, y_val in your environment
load('validationSet.mat');

fprintf('\nEvaluating the trained Linear SVM on a validation set ...\n')

p = svmPredict(model, X_val);

fprintf('Validation Accuracy: %f\n', mean(double(p == y_val)) * 100);
pause;



%% =================== Test Spam Classification ================
% Load the test dataset
% You will have X_test, y_test in your environment
load('testSet.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, X_test);

fprintf('Test Accuracy: %f\n', mean(double(p == y_test)) * 100);
pause;


%% =================  Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Try Your Own Emails =====================
filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

