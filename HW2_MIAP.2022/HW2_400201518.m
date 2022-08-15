%% 1)

% Load image
im1 = imread('melanome1.jpg');
im2 = imread('melanome2.jpg');
im3 = imread('melanome3.jpg');
im4 = imread('melanome4.jpg');

figure()
subplot(2,2,1)
imshow(im1)
title('melanome 1')

subplot(2,2,2)
imshow(im2)
title('melanome 2')

subplot(2,2,3)
imshow(im3)
title('melanome 3')

subplot(2,2,4)
imshow(im4)
title('melanome 4')

% Binarize
im1 = ~imbinarize(rgb2gray(im1),0.25);
im2 = ~imbinarize(rgb2gray(im2),0.25);
im3 = ~imbinarize(rgb2gray(im3),0.25);
im4 = ~imbinarize(rgb2gray(im4),0.25);

figure()
subplot(2,2,1)
imshow(im1)
title('melanome 1')

subplot(2,2,2)
imshow(im2)
title('melanome 2')

subplot(2,2,3)
imshow(im3)
title('melanome 3')

subplot(2,2,4)
imshow(im4)
title('melanome 4')

%% A)
im_con = imclose(im1,strel('disk',3));
figure()

subplot(1,2,1)
imshow(im1)
title('melanome 1 (orignal)')

subplot(1,2,2)
imshow(im_con)
title('melanome 1 (connected)')

bwconncomp(im_con)

%% B)
im2_op = imopen(im2, strel('disk',3));
im2_bond = im2_op - imerode(im2_op, strel('disk',3));

subplot(1,2,1)
imshow(im2)
title('melanome 2 (orignal)')

subplot(1,2,2)
imshow(im2_bond)
title('melanome 2 (bound after denoising)')

%% C)
im3_con = imclose(im3,strel('disk',3));
im3_fill = imfill(im3_con,'holes');

figure()

subplot(1,2,1)
imshow(im3)
title('melanome 3 (orignal)')

subplot(1,2,2)
imshow(im3_fill)
title('melanome 3 (filled)')

%% D)
im4_con = imopen(im4,strel('disk',17));

figure()

subplot(1,2,1)
imshow(im4)
title('melanome 4 (orignal)')

subplot(1,2,2)
imshow(im4_con)
title('melanome 4 (deperated)')

%% E)
im_out = im_cc(im4_con);
for i = 1:size(im_out,1)
    figure()
    imshow(squeeze(im_out(i,:,:)))
    title(['Segment ', num2str(i)])
end


%% 2)

%% A)

% Load image
im_brain = imread('brain.jpg');

figure()
imshow(im_brain)
title('brain.jpg')

hist_img = hist(reshape(im_brain, 1, []), 0:255);
norm_hist_img = hist_img/sum(hist_img);

m_brain = sum((0:255).*norm_hist_img,2)
var_brain = sum(((0:255)-m_brain).^2.*norm_hist_img,2)
uni_brain = sum(norm_hist_img.^2,2)
en_brain = -sum(norm_hist_img.*log2(norm_hist_img+eps),2)

%% B)

% Load image
im_sample = imread('sample.png');

figure()
imshow(im_sample)
title('sample.png')

hist_img = hist(reshape(im_sample, 1, []), 0:255);
norm_hist_img = hist_img/sum(hist_img);

m_sample = sum((0:255).*norm_hist_img,2)
var_sample = sum(((0:255)-m_sample).^2.*norm_hist_img,2)
uni_sample = sum(norm_hist_img.^2,2)
en_sample = -sum(norm_hist_img.*log2(norm_hist_img+eps),2)

%% C
glcm_brain = my_glcm(im_brain, 256, [0, 1]);
figure()
imagesc(log(glcm_brain+1))
title('Log ( brain GLCM )')

glcm_sample = my_glcm(im_sample, 256, [0, 1]);
figure()
imagesc(log(glcm_sample+1))
title('Log ( sample GLCM )')

%% D

plcm_brain = glcm_brain / sum(sum(glcm_brain));
plcm_sample = glcm_sample / sum(sum(glcm_sample));

Contrast_brain = 0;
Contrast_sample = 0;

for i=1:256 
    for j=1:256 
        Contrast_brain = Contrast_brain + (i-j)^2*plcm_brain(i,j); 
        Contrast_sample = Contrast_sample + (i-j)^2*plcm_sample(i,j); 
    end
end

Uniformity_brain = 0;
Uniformity_sample = 0;

for i=1:256 
    for j=1:256 
        Uniformity_brain = Uniformity_brain + plcm_brain(i,j)^2; 
        Uniformity_sample = Uniformity_sample + plcm_sample(i,j)^2; 
    end
end

Homogeneity_brain = 0;
Homogeneity_sample = 0;

for i=1:256 
    for j=1:256 
        Homogeneity_brain = Homogeneity_brain + plcm_brain(i,j)/(1+abs(i-j));
        Homogeneity_sample = Homogeneity_sample + plcm_sample(i,j)/(1+abs(i-j));
    end
end

Entropy_brain = 0;
Entropy_sample = 0;

for i=1:256 
    for j=1:256 
        Entropy_brain = Entropy_brain - plcm_brain(i,j)*log2(plcm_brain(i,j)+eps); 
        Entropy_sample = Entropy_sample - plcm_sample(i,j)*log2(plcm_sample(i,j)+eps); 
    end
end

fprintf('- Brain - \nContrast: %f \nUniformity: %f \nHomogeneity: %f\nEntropy: %f\n\n - Sample - \nContrast: %f \nUniformity: %f \nHomogeneity: %f\nEntropy: %f\n\n', ...
    Contrast_brain, Uniformity_brain, Homogeneity_brain, Entropy_brain,...
    Contrast_sample, Uniformity_sample, Homogeneity_sample, Entropy_sample)

%% E

glcm_brain = my_glcm(im_brain, 128, [0, 1]);
figure()
imagesc(log(glcm_brain+1))
title('Log ( brain GLCM )')

glcm_sample = my_glcm(im_sample, 128, [0, 1]);
figure()
imagesc(log(glcm_sample+1))
title('Log ( sample GLCM )')

plcm_brain = glcm_brain / sum(sum(glcm_brain));
plcm_sample = glcm_sample / sum(sum(glcm_sample));

Contrast_brain = 0;
Contrast_sample = 0;

for i=1:128 
    for j=1:128 
        Contrast_brain = Contrast_brain + (i-j)^2*plcm_brain(i,j); 
        Contrast_sample = Contrast_sample + (i-j)^2*plcm_sample(i,j); 
    end
end

Uniformity_brain = 0;
Uniformity_sample = 0;

for i=1:128 
    for j=1:128 
        Uniformity_brain = Uniformity_brain + plcm_brain(i,j)^2; 
        Uniformity_sample = Uniformity_sample + plcm_sample(i,j)^2; 
    end
end

Homogeneity_brain = 0;
Homogeneity_sample = 0;

for i=1:128 
    for j=1:128 
        Homogeneity_brain = Homogeneity_brain + plcm_brain(i,j)/(1+abs(i-j));
        Homogeneity_sample = Homogeneity_sample + plcm_sample(i,j)/(1+abs(i-j));
    end
end

Entropy_brain = 0;
Entropy_sample = 0;

for i=1:128 
    for j=1:128 
        Entropy_brain = Entropy_brain - plcm_brain(i,j)*log2(plcm_brain(i,j)+eps); 
        Entropy_sample = Entropy_sample - plcm_sample(i,j)*log2(plcm_sample(i,j)+eps); 
    end
end

fprintf('- Brain - \nContrast: %f \nUniformity: %f \nHomogeneity: %f\nEntropy: %f\n\n - Sample - \nContrast: %f \nUniformity: %f \nHomogeneity: %f\nEntropy: %f\n\n', ...
    Contrast_brain, Uniformity_brain, Homogeneity_brain, Entropy_brain,...
    Contrast_sample, Uniformity_sample, Homogeneity_sample, Entropy_sample)

%% 3)

im_boat = imread('boat.png');
figure()
imshow(im_boat)
title('boat.png')

%% B
Lp = (1/4/sqrt(2))*[1+sqrt(3), 3+sqrt(3), 3-sqrt(3), 1-sqrt(3)];
Hp = (1/4/sqrt(2))*[1-sqrt(3), -3+sqrt(3), 3+sqrt(3), -1-sqrt(3)];

% Convolve with filter X the rows of the entry
LoD = conv2(1,Lp,im_boat,'same');
HiD = conv2(1,Hp,im_boat,'same');

% Downsample columns: keep the even-indexed columns
LoD = LoD(:, 2:2:end);
HiD = HiD(:, 2:2:end);

% Convolve with filter X the columns of the entry
LoD_LoD = conv2(Lp,1,LoD,'same');
LoD_HiD = conv2(Hp,1,LoD,'same');

HiD_LoD = conv2(Lp,1,HiD,'same');
HiD_HiD = conv2(Hp,1,HiD,'same');

% Downsample rows: keep the even-indexed rows
LoD_LoD = LoD_LoD(2:2:end, :);
LoD_HiD = LoD_HiD(2:2:end, :);
HiD_LoD = HiD_LoD(2:2:end, :);
HiD_HiD = HiD_HiD(2:2:end, :);

figure()
subplot(2,2,1)
imshow(LoD_LoD/max(max(LoD_LoD)))

subplot(2,2,2)
imshow(LoD_HiD/max(max(LoD_HiD)))

subplot(2,2,3)
imshow(HiD_LoD/max(max(HiD_LoD)))

subplot(2,2,4)
imshow(HiD_HiD/max(max(HiD_HiD)))

%% C
Lp = (1/4/sqrt(2))*[3-sqrt(3), 3+sqrt(3), 1+sqrt(3), 1-sqrt(3)];
Hp = (1/4/sqrt(2))*[1-sqrt(3), -1-sqrt(3), 3+sqrt(3), -3+sqrt(3)];

% Upsample rows: insert zeros at odd-indexed rows
tmp = zeros(2*size(LoD_LoD,1),size(LoD_LoD,2));
tmp(2:2:end,:) = LoD_LoD;
LoD_LoD_x = tmp;
tmp = zeros(2*size(LoD_HiD,1),size(LoD_HiD,2));
tmp(2:2:end,:) = LoD_HiD;
LoD_HiD_x = tmp;
tmp = zeros(2*size(HiD_LoD,1),size(HiD_LoD,2));
tmp(2:2:end,:) = HiD_LoD;
HiD_LoD_x = tmp;
tmp = zeros(2*size(HiD_HiD,1),size(HiD_HiD,2));
tmp(2:2:end,:) = HiD_HiD;
HiD_HiD_x = tmp;

% Convolve with filter X the columns of the entry
X_lo = conv2(Lp,1,LoD_LoD_x,'same') + conv2(Hp,1,LoD_HiD_x,'same');
X_hi = conv2(Lp,1,HiD_LoD_x,'same') + conv2(Hp,1,HiD_HiD_x,'same');

% Upsample columns: insert zeros at odd-indexed columns
tmp = zeros(size(X_lo,1),2*size(X_lo,2));
tmp(:,2:2:end) = X_lo;
X_lo = tmp;
tmp = zeros(size(X_hi,1),2*size(X_hi,2));
tmp(:,2:2:end) = X_hi;
X_hi = tmp;

% Convolve with filter X the rows of the entry
Y = conv2(1,Lp,X_lo,'same') + conv2(1,Hp,X_hi,'same');
imshow(Y/max(max(Y)))

% RMSE
rmse = sqrt(mean((double(im_boat)-Y).^2,'all'))

%% D
Per = 0.1;
stack = sort([LoD_LoD(:); LoD_HiD(:); HiD_LoD(:); HiD_HiD(:)], 'descend');
thr = stack(round(Per*size(stack,1)));

Lp = (1/4/sqrt(2))*[3-sqrt(3), 3+sqrt(3), 1+sqrt(3), 1-sqrt(3)];
Hp = (1/4/sqrt(2))*[1-sqrt(3), -1-sqrt(3), 3+sqrt(3), -3+sqrt(3)];

LoD_LoD_temp = LoD_LoD .* (LoD_LoD>thr);
LoD_HiD_temp = LoD_HiD .* (LoD_HiD>thr);
HiD_LoD_temp = HiD_LoD .* (HiD_LoD>thr);
HiD_HiD_temp = HiD_HiD .* (HiD_HiD>thr);

% Upsample rows: insert zeros at odd-indexed rows
tmp = zeros(2*size(LoD_LoD_temp,1),size(LoD_LoD_temp,2));
tmp(2:2:end,:) = LoD_LoD_temp;
LoD_LoD_x = tmp;
tmp = zeros(2*size(LoD_HiD_temp,1),size(LoD_HiD_temp,2));
tmp(2:2:end,:) = LoD_HiD_temp;
LoD_HiD_x = tmp;
tmp = zeros(2*size(HiD_LoD_temp,1),size(HiD_LoD_temp,2));
tmp(2:2:end,:) = HiD_LoD_temp;
HiD_LoD_x = tmp;
tmp = zeros(2*size(HiD_HiD_temp,1),size(HiD_HiD_temp,2));
tmp(2:2:end,:) = HiD_HiD_temp;
HiD_HiD_x = tmp;

% Convolve with filter X the columns of the entry
X_lo = conv2(Lp,1,LoD_LoD_x,'same') + conv2(Hp,1,LoD_HiD_x,'same');
X_hi = conv2(Lp,1,HiD_LoD_x,'same') + conv2(Hp,1,HiD_HiD_x,'same');

% Upsample columns: insert zeros at odd-indexed columns
tmp = zeros(size(X_lo,1),2*size(X_lo,2));
tmp(:,2:2:end) = X_lo;
X_lo = tmp;
tmp = zeros(size(X_hi,1),2*size(X_hi,2));
tmp(:,2:2:end) = X_hi;
X_hi = tmp;

% Convolve with filter X the rows of the entry
Y = conv2(1,Lp,X_lo,'same') + conv2(1,Hp,X_hi,'same');
imshow(Y/max(max(Y)))

% RMSE
rmse = sqrt(mean((double(im_boat)-Y).^2,'all'))


%% E

FFT = fft2(im_boat);

Per = 0.1;
stack = sort(abs(FFT(:)), 'descend');
thr = stack(round(Per*size(stack,1)));

FFT = FFT .* (abs(FFT)>thr);

Y = ifft2(FFT);
imshow(abs(Y)/max(max(abs(Y))))

% RMSE
rmse = sqrt(mean((double(im_boat)-Y).^2,'all'))

%% 4)

im_covid = imread('covid.png');
im_covid = rgb2gray(im_covid);
figure()
imshow(im_covid)
title('covid.png')

%% A, B)
[LoD,HiD] = wfilters('haar','d');

[cA,cH,cV,cD] = dwt2(im_covid,LoD,HiD,'mode','symh');
subplot(2,2,1)
imagesc(cA)
colormap gray
title('Approximation')
subplot(2,2,2)
imagesc(cH)
colormap gray
title('Horizontal')
subplot(2,2,3)
imagesc(cV)
colormap gray
title('Vertical')
subplot(2,2,4)
imagesc(cD)
colormap gray
title('Diagonal')

th_ch = 0.1;
th_cv = th_ch;
th_cd = 0.1;

cH = cH .* cH>th_ch;
cV = cV .* cV>th_cv;
cD = cD .* cD>th_cd;

[LoR,HiR] = wfilters('haar','r');
Y = idwt2(cA,cH,cV,cD,LoR,HiR);

figure()
subplot(1,2,1)
imshow(im_covid)
title('Original')
subplot(1,2,2)
imshow(Y/max(max(Y)))
title('DWT Filtered')

% RMSE
rmse = sqrt(mean((double(im_covid)-Y).^2,'all'))

%% C)
FFT = fft2(im_covid);

Per = 0.3;
stack = sort(abs(FFT(:)), 'descend');
thr = stack(round(Per*size(stack,1)));

FFT = FFT .* (abs(FFT)>thr);

Y = ifft2(FFT);

figure()
subplot(1,2,1)
imshow(im_covid)
title('Original')
subplot(1,2,2)
imshow(real(Y)/max(max(real(Y))))
title('DFT Filtered')

% RMSE
rmse = sqrt(mean((double(im_covid)-Y).^2,'all'))

%% Function
function im_out = im_cc (im)

counter = 1;
im_out = zeros([counter,size(im)]);

while (max(max(im))) 
    im_temp = zeros(size(im));
    im_temp_new = zeros(size(im));

    [X,Y] = find(im==1);
    im_temp_new(X(1),Y(1)) = 1;
    
    while (max(max(im_temp ~= im_temp_new)))
        im_temp = im_temp_new;
        im_temp_new = imdilate(im_temp_new, strel('rectangle',[3,3])) & im;
    end
    
    im_out(counter, :, :) = im_temp_new;
    im = im - im_temp_new;
    counter = counter + 1;
    
end

end


function glcm = my_glcm(im, numLevels, offset)

    levs = ceil(linspace(-0.5,255.5,numLevels+1));
    glcm = zeros(numLevels, numLevels);
    
    x_start = round(max(-offset(1)-1,1));
    x_end = round(min(size(im,1) - offset(1),size(im,1)));
    y_start = round(max(-offset(2)-1,1));
    y_end = round(min(size(im,2) - offset(2),size(im,2)));
    
    for i = x_start:x_end
        for j = y_start:y_end
            x = find(im(i, j)<levs,1)-1;
            y = find(im(i+offset(1), j+offset(2))<levs,1)-1;
            glcm(x,y) = glcm(x,y) + 1;
        end   
    end
    
end
 
