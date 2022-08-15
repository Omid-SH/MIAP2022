% Image segmentation methods
% Auth : Omid Sharafi (2022)
% Git-Hub : https://github.com/Omid-SH

%% 1)

%% a)
mri(:,:,1) = imread('MRI1.png');
mri(:,:,2) = imread('MRI2.png');
mri(:,:,3) = imread('MRI3.png');
mri(:,:,4) = imread('MRI4.png');
mri = double(mri)/255;

figure()
imshow(mri(:,:,1:3))
title('Mix MRI 1,2,3')

figure()
imshow(mri(:,:,2:4))
title('Mix MRI 2,3,4')

%% b)

figure()
for i = 1:4
    subplot(4, 4, i)
    imshow(mri(:,:,i))
end

options = [1.1 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), 4, options);
fprintf('FCM :\n Fuzzy coefficient : %d\n Number of iteration : %d\n Objective Function : %d\n\n', 1.1, iter_n, obj_fcn(iter_n));
for i = 1:4
    subplot(4, 4, 4+i)
    imshow(reshape(U(i,:), 200, 256))
end

options = [2 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), 4, options);
fprintf('FCM :\n Fuzzy coefficient : %d\n Number of iteration : %d\n Objective Function : %d\n\n', 2, iter_n, obj_fcn(iter_n));
for i = 1:4
    subplot(4, 4, 8+i)
    imshow(reshape(U(i,:), 200, 256))
end

options = [5 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), 4, options);
fprintf('FCM :\n Fuzzy coefficient : %d\n Number of iteration : %d\n Objective Function : %d\n\n', 5, iter_n, obj_fcn(iter_n));
for i = 1:4
    subplot(4, 4, 12+i)
    imshow(reshape(U(i,:), 200, 256))
end

%% c)

figure()
image = uint8(mri.*255);

for cluster_n = 2:5
    [L,Centers] = imsegkmeans(image,cluster_n);
    subplot(2,2,cluster_n-1)
    imagesc(L)
    title(['Cluster Number = ', num2str(cluster_n)])
end

% FCM with initial condition
cluster_n = 4;
[L,Centers] = imsegkmeans(image,cluster_n);

L = double(reshape(L, 1, []));
L = double(ind2vec(L));

figure()
for i = 1:4
    subplot(4, 4, i)
    imshow(mri(:,:,i))
end

options = [1.1 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), cluster_n, options, L);
fprintf('FCM with initial condition:\n Fuzzy coefficient : %d\n Number of iteration : %d\n Objective Function : %d\n\n', 1.1, iter_n, obj_fcn(iter_n));
for i = 1:4
    subplot(4, 4, 4+i)
    imshow(reshape(U(i,:), 200, 256))
end

options = [2 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), cluster_n, options, L);
fprintf('FCM with initial condition:\n Fuzzy coefficient : %d\n Number of iteration : %d\n Objective Function : %d\n\n', 2, iter_n, obj_fcn(iter_n));
for i = 1:4
    subplot(4, 4, 8+i)
    imshow(reshape(U(i,:), 200, 256))
end

options = [5 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), cluster_n, options, L);
fprintf('FCM with initial condition:\n Fuzzy coefficient : %d\n Number of iteration : %d\n Objective Function : %d\n\n', 5, iter_n, obj_fcn(iter_n));
for i = 1:4
    subplot(4, 4, 12+i)
    imshow(reshape(U(i,:), 200, 256))
end

%% d)
GMM_model = fitgmdist(reshape(mri, [], 4), 4, 'RegularizationValue', 0.003);
indices = cluster(GMM_model, reshape(mri, [], 4));
imagesc(reshape(indices, 200, 256));
title('GMM Result');

%% e)
options = [2 NaN NaN 0];
[center, U, obj_fcn, iter_n] = fcm(reshape(mri, [], 4), cluster_n, options, L);
imshow(reshape(max(U) > max(1.25/4, 1/2), 200, 256))

%% 2)

%% a)
melanoma = imread('melanoma_gray.jpg');
nevus = imread('nevus_gray.jpg');

addpath('./snake_demo/snake')
addpath('./activeContoursSnakesDemo/activeContoursDemo/')

mu=0.1;
ITER=100;
[u,v] = GVF(melanoma, mu, ITER);
u = u ./ max(abs(u),[],'all');
v = v./ max(abs(u),[],'all');

figure()
subplot(1,2,1)
imshow(melanoma)
title('Orginal Image')

subplot(1,2,2)
imshow(labeloverlay(melanoma,((u.^2 + v.^2)>0.1)))
title('GVF Output')

figure()
try
quiver(u(end:-1:1, :),v(end:-1:1, :))
catch
end

[u,v] = GVF(nevus, mu, ITER);
u = u ./ max(abs(u),[],'all');
v = v./ max(abs(u),[],'all');

figure()
try
quiver(u(end:-1:1, :),v(end:-1:1, :))
catch
end

figure()
subplot(1,2,1)
imshow(nevus)
title('Orginal Image')

subplot(1,2,2)
imshow(labeloverlay(nevus,((u.^2 + v.^2)>0.1)))
title('GVF Output')

%% Basic Snake
snk

%% b)
image = mri(:,:,1);
[u,v] = GVF(image, mu, ITER);
u = u ./ max(abs(u),[],'all');
v = v./ max(abs(u),[],'all');

try
quiver(u(end:-1:1, :),v(end:-1:1, :))
catch
end

figure()
subplot(1,2,1)
imshow(image)
title('Orginal Image')

subplot(1,2,2)
imshow(labeloverlay(image,((u.^2 + v.^2)>0.01)))
title('GVF Output')

%% Basic Snake
snk

%% 3)

%% a, b)
addpath('./Chan-Vese')
% I = melanoma;
I = nevus;
% I = mri(:,:,3);

%% user mask
BW = roipoly(I);
seg = chenvese(I,BW,500,0.1,'chan');

%% user point
imshow(I);
[y,x] = ginput(1);
m = zeros(size(I,1),size(I,2));
m(max(x-4,1):min(x+4,size(I,1)),max(y-4,1):min((y+4),size(I,2))) = 1;
chenvese(I,m,500,0.1,'chan');


%% b)
%% Auto Segmentation Based on Lighter region

% find lighter 21*21 kernel
M = medfilt2(I,[21 21]);
[x, y] = find(M == max(M,[],'all'), 1);
m = zeros(size(I,1),size(I,2));
m(max(x-10,1):min(x+10,size(I,1)),max(y-10,1):min((y+10),size(I,2))) = 1;
chenvese(I,m,500,0.1,'chan');

%% Auto Segmentation
chenvese(I,'medium',400,0.02,'chan'); 

%% Functions

function [center, U, obj_fcn, iter_n] = fcm(data, cluster_n, options, U_in)
%FCM Data set clustering using fuzzy c-means clustering.
%
%   [CENTER, U, OBJ_FCN] = FCM(DATA, N_CLUSTER) finds N_CLUSTER number of
%   clusters in the data set DATA. DATA is size M-by-N, where M is the number of
%   data points and N is the number of coordinates for each data point. The
%   coordinates for each cluster center are returned in the rows of the matrix
%   CENTER. The membership function matrix U contains the grade of membership of
%   each DATA point in each cluster. The values 0 and 1 indicate no membership
%   and full membership respectively. Grades between 0 and 1 indicate that the
%   data point has partial membership in a cluster. At each iteration, an
%   objective function is minimized to find the best location for the clusters
%   and its values are returned in OBJ_FCN.
%
%   [CENTER, ...] = FCM(DATA,N_CLUSTER,OPTIONS) specifies a vector of options
%   for the clustering process:
%       OPTIONS(1): exponent for the matrix U             (default: 2.0)
%       OPTIONS(2): maximum number of iterations          (default: 100)
%       OPTIONS(3): minimum amount of improvement         (default: 1e-5)
%       OPTIONS(4): info display during iteration         (default: 1)
%   The clustering process stops when the maximum number of iterations
%   is reached, or when the objective function improvement between two
%   consecutive iterations is less than the minimum amount of improvement
%   specified. Use NaN to select the default value.
%
%   Example
%       data = rand(100,2);
%       [center,U,obj_fcn] = fcm(data,2);
%       plot(data(:,1), data(:,2),'o');
%       hold on;
%       maxU = max(U);
%       % Find the data points with highest grade of membership in cluster 1
%       index1 = find(U(1,:) == maxU);
%       % Find the data points with highest grade of membership in cluster 2
%       index2 = find(U(2,:) == maxU);
%       line(data(index1,1),data(index1,2),'marker','*','color','g');
%       line(data(index2,1),data(index2,2),'marker','*','color','r');
%       % Plot the cluster centers
%       plot([center([1 2],1)],[center([1 2],2)],'*','color','k')
%       hold off;
%
%   See also FCMDEMO, INITFCM, IRISFCM, DISTFCM, STEPFCM.

%   Roger Jang, 12-13-94, N. Hickey 04-16-01
%   Copyright 1994-2018 The MathWorks, Inc. 

if nargin ~= 2 && nargin ~= 3 && nargin ~= 4
	error(message("fuzzy:general:errFLT_incorrectNumInputArguments"))
end

data_n = size(data, 1);

% Change the following to set default options
default_options = [2;	% exponent for the partition matrix U
		100;	% max. number of iteration
		1e-5;	% min. amount of improvement
		1];	% info display during iteration 

if nargin == 2
	options = default_options;
else
	% If "options" is not fully specified, pad it with default values.
	if length(options) < 4
		tmp = default_options;
		tmp(1:length(options)) = options;
		options = tmp;
	end
	% If some entries of "options" are nan's, replace them with defaults.
	nan_index = find(isnan(options)==1);
	options(nan_index) = default_options(nan_index);
	if options(1) <= 1
		error(message("fuzzy:general:errFcm_expMustBeGtOne"))
	end
end

expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);		% Display info or not

obj_fcn = zeros(max_iter, 1);	% Array for objective function

%--------------------------------------------------------------------------
if nargin == 3
    U = initfcm(cluster_n, data_n);			% Initial fuzzy partition
else
    U = U_in;
end
%--------------------------------------------------------------------------

% Main loop
for i = 1:max_iter
	[U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
	if display
		fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
	end
	% check termination condition
	if i > 1
		if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end
	end
end

iter_n = i;	% Actual number of iterations 
obj_fcn(iter_n+1:max_iter) = [];

end