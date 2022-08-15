%% P1
%% A

img = rgb2gray(imread('histogram.jpeg'));
hist_img = hist(reshape(img, 1, []), 0:255);
hist_img = hist_img/(size(img, 1)*size(img, 2));

map_img = zeros(256, 1);
sum = 0;
for i=1:256
    sum = sum + hist_img(i);
    map_img(i) = round(sum*255);
end

img_equalized = img;
for i=1:size(img, 1)
   for j=1:size(img, 2)
       img_equalized(i, j) = map_img(img(i, j)+1);
   end
end

hist_img_equalized = hist(reshape(img_equalized, 1, []), 0:255);
hist_img_equalized = hist_img_equalized/(size(img_equalized, 1)*size(img_equalized, 2));

figure();
subplot(3, 2, [1, 3]);
imshow(img);
title('Image');

subplot(3, 2, [2, 4]);
imshow(img_equalized);
title('Equalized Image');

subplot(3, 2, 5);
bar(0:255, hist_img);
title('Histogram of Image');

subplot(3, 2, 6);
bar(0:255, hist_img_equalized);
title('Histogram of Equalized Image');

%% B

my_hist_img = my_hist(img);
my_hist_img_equalized = my_hist(img_equalized);

figure();
subplot(3, 2, [1, 3]);
imshow(img);
title('Image');

subplot(3, 2, [2, 4]);
imshow(img_equalized);
title('Equalized Image');

subplot(3, 2, 5);
bar(0:255, my_hist_img);
title('Histogram of Image (using my-hist function)');

subplot(3, 2, 6);
bar(0:255, my_hist_img_equalized);
title('Histogram of Equalized Image (using my-hist function)');

%% C
my_hist_img = my_hist(img);
my_hist_img_equalized = my_hist(img_equalized);

[img_local_equalized, hist_img_local_equalized] = local_hist(img);

figure();
subplot(3, 3, [1, 4]);
imshow(img);
title('Image');

subplot(3, 3, [2, 5]);
imshow(img_equalized);
title('Equalized Image');

subplot(3, 3, [3, 6]);
imshow(img_local_equalized);
title('Local Equalized Image');

subplot(3, 3, 7);
bar(0:255, my_hist_img);
title('Histogram of Image (using my-hist function)');

subplot(3, 3, 8);
bar(0:255, my_hist_img_equalized);
title('Histogram of Equalized Image (using my-hist function)');

subplot(3, 3, 9);
bar(0:255, hist_img_local_equalized);
title('Histogram of Local Equalized Image (using my-hist function)');

%% P2

%% A
% load original image
input = imread('./brainMRI.png');
input = rgb2gray(input);
figure()
imshow(input)
title('Original Image')

% add noise to image
midx = round(size(input,1)/2);
midy = round(size(input,2)/2);
noisy = input;

noisy(1:midx,1:midy) = imnoise(input(1:midx,1:midy),'salt & pepper',0.01);
noisy(midx+1:end,1:midy) = imnoise(input(midx+1:end,1:midy),'gaussian', 0, 0.05);
noisy(midx+1:end,midy+1:end) = imnoise(imnoise(input(midx+1:end,midy+1:end),'gaussian', 0, 0.05),'salt & pepper',0.01);

figure()
imshow(noisy)
title('Noisy Image')

%% B
% Denoising using filters
avg_out = imfilter(noisy, fspecial('average',5));
med_out = medfilt2(noisy,[5 5]);
gauss_out = imfilter(noisy, fspecial('gaussian',5,1));

figure()
subplot(2,2,1)
imshow(noisy)
title('Noisy Image')

subplot(2,2,2)
imshow(avg_out)
title('Average Filter Output')

subplot(2,2,3)
imshow(med_out)
title('Median Filter Output')

subplot(2,2,4)
imshow(gauss_out)
title('Gaussian Filter Output')

%% C 
% uncomment noise adding part you want and then run above code box

% noisy = imnoise(imnoise(input,'gaussian', 0, 0.05),'salt & pepper',0.01);
noisy = imnoise(imnoise(input,'salt & pepper',0.01),'gaussian', 0, 0.05);

%% Add new conmbined filter (median + gaussian) 

gaussian_med_out = imfilter(medfilt2(noisy,[5 5]), fspecial('gaussian',5,1));
med_gaussian_out = medfilt2(imfilter(noisy, fspecial('gaussian',5,1)),[5 5]);

figure()
subplot(1,3,1)
imshow(noisy)
title('Noisy Image')

subplot(1,3,2)
imshow(gaussian_med_out)
title('Gaussian(Median) Output')

subplot(1,3,3)
imshow(med_gaussian_out)
title('Median(Gaussian) Output')

%% D
wiener_out = wiener2(noisy,[5 5],0.1);

figure()
subplot(1,2,1)
imshow(noisy)
title('Noisy Image')

subplot(1,2,2)
imshow(wiener_out)
title('Wiener Filter Output')

%% P3
%% A
img = rgb2gray(imread('wall.jpeg'));

figure();
subplot(2,3,1)

imshow(img);
title('image');

subplot(2,3,2)
img_filtered = imfilter(img, [1, -1]);
imshow(img_filtered);
title('Filtered Image using [1, -1]');

subplot(2,3,3)
img_filtered = imfilter(img, [1, 0, -1]);
imshow(img_filtered);
title('Filtered Image using [1, 0, -1]');

subplot(2,3,4)
img_filtered = imfilter(img, [1, 0, -1]');
imshow(img_filtered);
title("Filtered Image using [1, 0, -1]'");

subplot(2,3,5)
img_filtered = imfilter(img, [1, -2, 1]);
imshow(img_filtered);
title('Filtered Image using [1, -2, 1]');

subplot(2,3,6)
img_filtered = imfilter(img, [1, -2, 1]');
imshow(img_filtered);
title("Filtered Image using [1, -2, 1]'");

%% B
figure();
subplot(1,2,1)
img_filtered = edge(img, 'Sobel');
imshow(img_filtered);
title('Filtered Image using Sobel filter');

subplot(1,2,2)
img_filtered = edge(img, 'Canny', 0.4);
imshow(img_filtered);
title('Filtered Image using Canny filter');

%% C
figure();
img_filtered = imfilter(img, [-1, -1, -1; -1, 8, -1; -1, -1, -1]);
imshow(img_filtered);
title('Filtered Image using laplacian');

%% D
figure();
img_filtered = medfilt2(imfilter(img, [0, 0, -1, 0, 0; 0, -1, -2, -1, 0; -1, -2, 16, -2, -1; 0, -1, -2, -1, 0;  0, 0, -1, 0, 0]), [5, 5]);
imshow(img_filtered);
title('Filtered Image using LoG');

%% P4
%% A, B
foot = imread('./foot.jpg');
hand = imread('./hand.jpg');

foot = rgb2gray(foot);
hand = rgb2gray(hand);

foot_f = fftshift(fft2(foot));
hand_f = fftshift(fft2(hand));

figure()
subplot(1,3,1)
imshow(foot)
title('Original Image')

subplot(1,3,2)
imshow(abs(foot_f)/max(max(abs(foot_f))))
title('fftshift Output')

subplot(1,3,3)
imshow(log10(abs(foot_f)+1)/max(max(log10(abs(foot_f)+1))))
title('log(fftshift) Output')

figure()
subplot(1,3,1)
imshow(hand)
title('Original Image')

subplot(1,3,2)
imshow(abs(hand_f)/max(max(abs(hand_f))))
title('fftshift Output')

subplot(1,3,3)
imshow(log10(abs(hand_f)+1)/max(max(log10(abs(hand_f)+1))))
title('log(fftshift) Output')

%% C
foot_abs = abs(foot_f);
foot_angle = angle(foot_f);

hand_abs = abs(hand_f);
hand_angle = angle(hand_f);

foot_hand = ifft2(foot_abs.*(cos(hand_angle)+sin(hand_angle).*1j));
hand_foot = ifft2(hand_abs.*(cos(foot_angle)+sin(foot_angle).*1j));

figure()
subplot(1,2,1)

imshow(abs(foot_hand)/max(max(abs(foot_hand))))
title('Foot abs + Hand Angle')

subplot(1,2,2)
imshow(abs(hand_foot)/max(max(abs(hand_foot))))
title('Hand abs + Foot Angle')

%% P5
% A
num_img = 6;
imds = imageDatastore('./river');
img = rgb2gray(readimage(imds,1));
data = zeros(length(imds.Files),numel(img));

figure()
for i = 1:length(imds.Files)
  img = rgb2gray(readimage(imds,i));
  data(i,:) = reshape(img,1,[]);
  subplot(2,3,i)
  imshow(img)
  title(['river ', num2str(i)])
end
data = data';
covv = cov(data);

[U, d] = eig(covv);
d = d(num_img:-1:1, num_img:-1:1);
U = U(:, 6:-1:1);

data_white = (data - mean(data)) * U;

cov(data_white)

figure()
for i = 1:num_img
    subplot(2,3,i)
    imshow(reshape(data_white(:,i)',360,360)/max(data_white(:,i)))
    title(['river ', num2str(i),' white'])
end

%% select first 2 eig

eig_num = 2;

crop_data_white = data_white(:,1:eig_num);
U_crop = U(:,1:eig_num)';
output_crop = crop_data_white * U_crop + mean(data);

figure()
for i = 1:num_img
    subplot(2,3,i)
    imshow(reshape(output_crop(:,i)',360,360)/max(output_crop(:,i)'))
    title(['river ', num2str(i),' output'])
end

%% functions

% functions
function hist_img = my_hist(img)
    hist_img = zeros(256, 1);
    for i=1:size(img, 1)
       for j=1:size(img, 2)
           hist_img(img(i, j)+1) = hist_img(img(i, j)+1)+1;
       end
    end
    hist_img = hist_img/(size(img, 1)*size(img, 2));
end

function [img_local_equalized, hist_img_local_equalized] = local_hist(img)
    img_local_equalized = img;
    field_view = 200;
    
    for i=1:size(img, 1)
       for j=1:size(img, 2)
           temp_min_x = max(1, i-field_view);
           temp_max_x = min(i+field_view, size(img, 1));
           temp_min_y = max(1, j-field_view);
           temp_max_y = min(j+field_view, size(img, 2));
           img_local_equalized(i, j) = round(length(find(img(temp_min_x:temp_max_x, temp_min_y:temp_max_y) <= img(i, j)))/((temp_max_x-temp_min_x+1)*(temp_max_y-temp_min_y+1))*255);
       end
    end
    
    hist_img_local_equalized = hist(reshape(img_local_equalized, 1, []), 0:255);
    hist_img_local_equalized = hist_img_local_equalized/(size(img_local_equalized, 1)*size(img_local_equalized, 2));
end