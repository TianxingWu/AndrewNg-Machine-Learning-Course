%% ===========Initialize================ 
clear;close all;clc;

%% ===========Load and Process data===============
directory_ham = 'D:\Users\James Wu\Desktop\work\ml\machine-learning-ex6\optional exercise\data_ham\';
directory_spam = 'D:\Users\James Wu\Desktop\work\ml\machine-learning-ex6\optional exercise\data_spam\';

X_ham = getFeatures(directory_ham);
X_spam = getFeatures(directory_spam);

% save X_ham.mat X_ham
% save X_spam.mat X_spam

%% =========Make set=============
% load ('X_ham.mat');
% load('X_spam.mat');

randIndex_ham = randperm(size(X_ham, 1));			% 生成1~m的随机序列
randIndex_spam = randperm(size(X_spam, 1));

X_ham_train = X_ham(randIndex_ham(1:2202), :);
X_ham_val = X_ham(randIndex_ham(2203:2936), :);
X_ham_test = X_ham(randIndex_ham(2937:3670), :);

X_spam_train = X_spam(randIndex_spam(1:900), :);
X_spam_val = X_spam(randIndex_spam(901:1200), :);
X_spam_test = X_spam(randIndex_spam(1201:1500), :);

X_train = [X_ham_train; X_spam_train];
X_val = [X_ham_val; X_spam_val];
X_test = [X_ham_test; X_spam_test];

y_train = [zeros(2202,1); ones(900,1)];
y_val = [zeros(734,1); ones(300,1)];
y_test = [zeros(734,1); ones(300,1)];

save trainSet.mat X_train y_train;
save validationSet.mat X_val y_val;
save testSet.mat X_test y_test;
