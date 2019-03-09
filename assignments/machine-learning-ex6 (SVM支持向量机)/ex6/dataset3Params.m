function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
i = 1;
C_sigmas = zeros(length(C)*length(sigma),2);
error = zeros(length(C)*length(sigma), 1);
for C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        error(i) = mean(double(predictions ~= yval));
        C_sigmas(i,:) = [C, sigma];
        i = i+1;      
    end
end

%fprintf('the best prameters are in %d row of "C_sigmas".\n',find(error == min(error)));
%fprintf('C_sigmas:\n');
%disp(C_sigmas);
%fprintf('\n');

best_C = C_sigmas(find(error == min(error)),1);
best_sigma = C_sigmas(find(error == min(error)),2);
fprintf('the best parameters are:\nC = %f, sigma = %f\n',best_C, best_sigma);
C = best_C;
sigma = best_sigma;
% =========================================================================

end
