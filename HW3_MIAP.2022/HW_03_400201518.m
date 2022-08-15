%% 1)
hand = imread('hand.jpg');
hand = imresize(hand, 0.2);
imshow(hand)

%% A)
hand_noisy = rgb2gray(imnoise(hand, 'gaussian', 0.05, 0.01));
hand = rgb2gray(hand);
montage([hand, hand_noisy])
hand = double(hand)/255;
hand_noisy = double(hand_noisy)/255;

% SNR
remained_noise = hand_noisy - hand;
SNR = 10*log10(sum(hand.^2,'all') / sum(remained_noise.^2,'all'))

%% B)
hx = 1;
n = size(hand_noisy,1);
m = size(hand_noisy,2);
hand_denoised_crf = zeros(n,m);

for x = 1:n
    for y = 1:m
        W = zeros(n, m);
        for xx = 1:n
            for yy = 1:m
                W(xx,yy) = exp(-((xx-x).^2 + (yy-y).^2)/(2*hx^2));
            end
        end
        hand_denoised_crf(x,y) = max(0, min(1, sum(W .* hand_noisy, 'all')/sum(W, 'all')));
    end
    if(mod(x,10) == 0)
        fprintf('%d/%d\n', x/10, floor(n/10))
    end
end

imshow(hand_denoised_crf)

% SNR
remained_noise = hand_noisy - hand_denoised_crf;
SNR = 10*log10(sum(hand.^2,'all') / sum(remained_noise.^2,'all'))

%% C)
hx = 1;
hg = 0.03;
n = size(hand,1);
m = size(hand,2);
hand_denoised_bf = zeros(n,m);

for x = 1:n
    for y = 1:m
        W = zeros(n, m);
        for xx = 1:n
            for yy = 1:m
                W(xx,yy) = exp(-((xx-x).^2 + (yy-y).^2)/(2*hx^2)).*exp(-(hand_noisy(xx,yy)-hand_noisy(x,y)).^2)/(2*hg^2);
            end
        end
        hand_denoised_bf(x,y) = max(0, min(1, sum(W .* hand_noisy, 'all')/sum(W, 'all')));
    end
    if(mod(x,10) == 0)
        fprintf('%d/%d\n', x/10, floor(n/10))
    end
end

imshow(hand_denoised_bf)

% SNR
remained_noise = hand_noisy - hand_denoised_bf;
SNR = 10*log10(sum(hand.^2,'all') / sum(remained_noise.^2,'all'))

%% D)
hand_denoised_nlm = NLM(hand_noisy, 2);
imshow(hand_denoised_nlm)

% SNR
remained_noise = hand_noisy - hand_denoised_nlm;
SNR = 10*log10(sum(hand.^2,'all') / sum(remained_noise.^2,'all'))

%% E)
% The location of the BM3D files -- this folder only contains demo data
addpath('bm3d');

% Call BM3D With the default settings.
y_est = BM3D(hand_noisy, 0.08);

figure
subplot(1, 3, 1);
imshow(hand);
title('Originall image');
subplot(1, 3, 2);
imshow(hand_noisy);
title('Noisy image');
subplot(1, 3, 3);
imshow(y_est);
title('BM3D denoised');

remained_noise = hand - y_est;
SNR = 10*log10(sum(hand.^2,'all') / sum(remained_noise.^2,'all'))

%% 2)

%% A)
p = phantom('Modified Shepp-Logan', 700);
p_noise = imnoise(p,'salt & pepper',0.03);

figure
subplot(1,2,1)
imshow(p)
title('original phantom')

subplot(1,2,2)
imshow(p_noise)
title('noisy phantom')

%% B)
p_denoised = anisodiff(p_noise, 5, 20, 0.11, 2);

figure
subplot(1,2,1)
imshow(p_noise)
title('noisy phantom')

subplot(1,2,2)
imshow(p_denoised)
title('denoised phantom')

%% C)
% EPI
H = fspecial('laplacian',0.2) ;
deltas = imfilter(p,H);
meandeltas = mean2(deltas);

deltascap = imfilter(p_denoised,H);
meandeltascap = mean2(deltascap);

p1 = deltas-meandeltas;
p2 = deltascap-meandeltascap;
num = sum(p1.*p2,'all');

den = sum(p1.^2,'all')*sum(p2.^2,'all');
EPI = num/sqrt(den)

% SNR
remained_noise = p_denoised - p;
SNR = 10*log10(sum(p.^2,'all') / sum(remained_noise.^2,'all'))

%% 3)

%% B)
p_denoised = TVL1denoise(p_noise, 1, 100);

figure
subplot(1,2,1)
imshow(p_noise)
title('noisy phantom')

subplot(1,2,2)
imshow(p_denoised)
title('denoised phantom')

%% C)
% EPI
H = fspecial('laplacian',0.2) ;
deltas = imfilter(p,H);
meandeltas = mean2(deltas);

deltascap = imfilter(p_denoised,H);
meandeltascap = mean2(deltascap);

p1 = deltas-meandeltas;
p2 = deltascap-meandeltascap;
num = sum(p1.*p2,'all');

den = sum(p1.^2,'all')*sum(p2.^2,'all');
EPI = num/sqrt(den)

% SNR
remained_noise = p_denoised - p;
SNR = 10*log10(sum(p.^2,'all') / sum(remained_noise.^2,'all'))

%% Function
function hand_denoised_nlm = NLM(hand_noisy, k)
    hv = 0.03 * (2*k+1);
    [n,m] = size(hand_noisy);
    hand_noisy_paded = padarray(hand_noisy,[k k],'symmetric');
    hand_denoised_nlm = zeros(n,m);

    for x = 1:n
        for y = 1:m
            Sg_x = hand_noisy_paded(x:(x+2*k), y:(y+2*k));
            W = zeros(n, m);
            for xx = 1:n
                for yy = 1:m
                    Sg_y = hand_noisy_paded(xx:(xx+2*k), yy:(yy+2*k));
                    W(xx,yy) = exp(-norm(Sg_x-Sg_y,'fro')/(2*hv^2));
                end
            end
            hand_denoised_nlm(x,y) = max(0, min(1, sum(W .* hand_noisy, 'all')/sum(W, 'all')));
        end
        if(mod(x,10) == 0)
            fprintf('%d/%d\n', x/10, floor(n/10))
        end
    end  
end