%% =============== Initialization ==============
clear ; close all; clc


%% ============= Part 4: K-Means Clustering on Pixels ===============
fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

goslin = double(imread('goslin.jpg'));
A = imresize(goslin, 1/5);

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run your K-Means algorithm on this data
% You should try different values of K and max_iters here
K1 = 10; 
K2 = 5;
K3 = 2;
max_iters = 10;

% When using K-Means, it is important the initialize the centroids
% randomly. 
% You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids1 = kMeansInitCentroids(X, K1);
initial_centroids2 = kMeansInitCentroids(X, K2);
initial_centroids3 = kMeansInitCentroids(X, K3);

% Run K-Means
pre_dist = 0;
for i = 1:50
    
    [centroids1, idx1] = runkMeans(X, initial_centroids1, max_iters);
    
    dist = 0;
    
    for j = 1:K1
        temp_idx = find(idx1 == j);
        temp = bsxfun(@minus, X(temp_idx), centroids1(j, :));
        temp = sum(sum(temp.^2, 2));
        dist = dist + temp;
    end
    
    if (dist>pre_dist && i~=1)
        centroids1 = pre_centroids1;
    end
    
    pre_centroids1 = centroids1;
    pre_dist = dist;
end


pre_dist = 0;
for i = 1:50
    
    [centroids2, idx2] = runkMeans(X, initial_centroids2, max_iters);
    
    dist = 0;
    
    for j = 1:K2
        temp_idx = find(idx2 == j);
        temp = bsxfun(@minus, X(temp_idx), centroids2(j, :));
        temp = sum(sum(temp.^2, 2));
        dist = dist + temp;
    end
    
    if (dist>pre_dist && i~=1)
        centroids2 = pre_centroids2;
    end
    
    pre_centroids2 = centroids2;
    pre_dist = dist;
end

pre_dist = 0;
for i = 1:50
    
    [centroids3, idx3] = runkMeans(X, initial_centroids3, max_iters);
    
    dist = 0;
    
    for j = 1:K3
        temp_idx = find(idx3 == j);
        temp = bsxfun(@minus, X(temp_idx), centroids3(j, :));
        temp = sum(sum(temp.^2, 2));
        dist = dist + temp;
    end
    
    if (dist>pre_dist && i~=1)
        centroids3 = pre_centroids3;
        dist = pre_dist;
    end
    
    pre_centroids3 = centroids3;
    pre_dist = dist;
end


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 5: Image Compression ======================
%  In this part of the exercise, you will use the clusters of K-Means to
%  compress an image. To do this, we first find the closest clusters for
%  each example. After that, we 

fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx1 = findClosestCentroids(X, centroids1);
idx2 = findClosestCentroids(X, centroids2);
idx3 = findClosestCentroids(X, centroids3);

% Essentially, now we have represented the image X as in terms of the
% indices in idx. 

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered1 = centroids1(idx1,:);
X_recovered2 = centroids2(idx2,:);
X_recovered3 = centroids3(idx3,:);

% Reshape the recovered image into proper dimensions
X_recovered1 = reshape(X_recovered1, img_size(1), img_size(2), 3);
X_recovered2 = reshape(X_recovered2, img_size(1), img_size(2), 3);
X_recovered3 = reshape(X_recovered3, img_size(1), img_size(2), 3);

% Display the original image 
subplot(2, 2, 1);
imagesc(A); 
axis image
title('Original Ryan Goslin');

% Display compressed image side by side
subplot(2, 2, 2);
imagesc(X_recovered1);
axis image
title(sprintf('Compressed, with %d colors.', K1));

subplot(2, 2, 3);
imagesc(X_recovered2);
axis image
title(sprintf('Compressed, with %d colors.', K2));

subplot(2, 2, 4);
imagesc(X_recovered3);
axis image
title(sprintf('Compressed, with %d colors.', K3));



fprintf('Program paused. Press enter to continue.\n');
pause;