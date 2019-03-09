function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

indexPos = find(y);
indexNeg = find(~y);

plot(X(indexPos, 1), X(indexPos, 2), 'k+', 'LineWidth', 2);
plot(X(indexNeg, 1), X(indexNeg, 2),  'ko', 'MarkerFaceColor', 'y');







% =========================================================================



hold off;

end
