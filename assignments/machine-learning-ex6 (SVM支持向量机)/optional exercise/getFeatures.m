function [ X ] = getFeatures( direc )
%GETFEATURES 此处显示有关此函数的摘要
%   此处显示详细说明

%% ===========Load and Process data===============
% direc = 'D:\Users\James Wu\Desktop\work\ml\machine-learning-ex6\optional exercise\data_ham\';
file_info = dir([direc, '*.txt']);

m = length(file_info);

file_contents = cell(m,1);
word_indices = file_contents;
x = file_contents;

fprintf('Loading and processing...\n');

for i = 1:m
   file_contents{i} = readFile([direc, file_info(i).name]);
   word_indices{i}  = processEmail(file_contents{i});
   x{i} = emailFeatures(word_indices{i});
   
   fprintf('Current number:\t%d\t\tTotal number:\t%d\n',i,m);
end

X = zeros(length(x), length(x{1}));
for k = 1:length(x)
X(k,:) = x{k};
end

fprintf('\nFinish！\n');


end

