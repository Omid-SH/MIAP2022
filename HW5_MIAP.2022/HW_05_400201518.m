% MIAP HW5-1
% Auth : Omid Sharafi - 2022
% Git-Hub : https://github.com/Omid-SH

%% 1
addpath('BCFCM/')

%%
Y = double(imread('test_biasfield_noise.png'))/255;
imshow(Y)

%%
Options.epsilon = 0.01;
Options.alpha = 0.85;
Options.p = 2;

[B,U]=BCFCM2D(Y, [0.42,0.51,0.69]', Options);

subplot(2,3,1)
imshow(Y)
title('Recorded Image')
subplot(2,3,2)
imshow(B)
title('Bias Field')
subplot(2,3,3)
imshow(Y - B)
title('Corrected Image')


for i = 1:3
    subplot(2,3,i+3)
    imshow(U(:,:,i))
    title(['U_',num2str(i)])
end

%% 2
addpath('Snake_GVF/')
addpath('Snake_GVF/Snake_GVF')

%%
I = double(imread('example.png'))/255;
[~,xi2,yi2] = roipoly(I);

%%
Options.nPoints = 100;
Options.Iterations = 500;

Options.Verbose = 0;
Options.Alpha = 0.1;
Options.Beta = 3;
Options.Delta = 0.4;
Options.Kappa = 4;

Options.Wline = 0.01;
Options.Wedge = 4.0;
Options.Wterm = 3;
[O,J] = Snake2D(I,[yi2 xi2], Options);

Irgb(:,:,1)=I;
Irgb(:,:,2)=I;
Irgb(:,:,3)=J;
Irgb = double(Irgb);
figure, imshow(Irgb,[]); 
hold on; 
plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);

%% 
Options.nPoints = 100;
Options.Iterations = 500;

Options.Verbose = 1;
Options.Alpha = 0.1;
Options.Beta = 3;
Options.Delta = 0.4;
Options.Kappa = 4;

Options.Wline = 0.01;
Options.Wedge = 4.0;
Options.Wterm = 3;
[O,J] = Snake2D(I,[yi2 xi2], Options);



Irgb(:,:,1)=I;
Irgb(:,:,2)=I;
Irgb(:,:,3)=J;
Irgb = double(Irgb);

figure, imshow(Irgb,[]); 
hold on; 
plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);