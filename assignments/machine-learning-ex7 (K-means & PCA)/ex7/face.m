%% Initialization
clear ; close all; clc


%% =============== Part 4: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');


pic = imread('go_orig.jpg');
B = imread('go.jpg');
A = rgb2gray(B);
A = double(A - mean(mean(A)));
img_size = size(A);
X_test = reshape(A, 1, img_size(1) * img_size(2));



%  Load Face dataset
load ('ex7faces.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);
[X_test_norm, mu_test, sigma_test] = featureNormalize(X_test);


%  Run PCA
[U, S] = pca(X_norm);

%  Visualize the top 36 eigenvectors found
figure;
displayData(U(:, 1:36)');
title(sprintf('人脸主成分特征向量(36 of 1024)'))

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 60;
Z = projectData(X_norm, U, K);


Z_test = projectData(X_test_norm, U, K);



fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 60;
X_rec  = recoverData(Z, U, K);



X_test_rec = recoverData(Z_test, U, K);



% Display normalized data
figure;

subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;





figure;

subplot(2,2,1);
imshow(pic);
title('1.高司令脸');
axis square;

subplot(2,2,2);
imshow(B);
title('2.高司令脸的缩略图');
axis square;

subplot(2,2,3);
displayData(X_test_norm);
title('3.高司令脸的缩略图的灰度图');
axis square;

subplot(2, 2, 4);
displayData(X_test_rec);
per = sum(sum((X_test_norm - X_test_rec).^2, 2))/sum(sum(X_test_norm.^2, 2));
title(sprintf('4.高司令脸的主成分(%2.2f%%variance)',100-per*100));
axis square;




fprintf('Program paused. Press enter to continue.\n');
pause;

