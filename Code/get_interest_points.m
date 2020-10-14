% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)
% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
%{
% for extra credit.
img = im2single(imread(image));     %change image to grayScaled
%===================Harris Coner Detector=======================
%1. Compute M matrix for each image window to get their cornerness scores
% --note: we can find M purely from the per-pixel image derivatives
%reduced high frequncy 
mean_mask = [1 1 1; 1 1 1; 1 1 1];
filtering = imfilter(image, mean_mask);

x_filter = [-1 0 1; -1 0 1; -1 0 1];
y_filter = [-1 -1 -1; 0 0 0; 1 1 1];


x_derivative = imfilter(filtering, x_filter);
y_derivative = imfilter(filtering, y_filter);
xy_derivative = sqrt(x_derivative.^2 + y_derivative.^2);

gaussian = fspecial('gaussian', feature_width);

%2. Find points whose surrounding window gave large corner response (f >
%threshold)
%3. Take the points of local maxima
%}
img = imread('image/4191453057_c86028ce1f_o.jpg')
img = img(:,:,1);

sigma = 1;
radius = 1;
order = (2*radius +1)^2;
threshold = 3000;

%derivatives in x and y direction
dx = [-1 0 1; -1 0 1; -1 0 1];
dy = [-1 -1 -1; 0 0 0; 1 1 1];
%[dx, dy] = meshgrid(-1:1, -1:1);

Ix = conv2(double(img), dx, 'same');
Iy = conv2(double(img), dy, 'same');

%% implementing the Gaussian filter

dim = max(1, fix(6*sigma));
m = dim; n=dim;

[h1, h2] = meshgrid(-(m-1)/2: (m-1)/2, -(n-1)/2: (n-2)/2);
hg = exp(-(h1.^2+h2.^2)/(2*sigma^2));
[a,b] = size(hg);
sum = 0;
for i = 1:a
    for j = 1:b
        sum = sum + hg(i,j);
    end
end

g = hg ./sum;


%Calculate entries of the M matrix
Ix2 = conv2(double(Ix.^2), g, 'same');
Iy2 = conv2(double(Iy.^2), g, 'same');
Ixy = conv2(double(Ix.*Iy), g, 'same');

%Harris measure
R = (Ix2.*Iy2 - Ixy.^2)./(Ix2+Iy2+eps);
mx = ordfilt2(R, order^2, ones(order));

harris_points = (R==mx) & (R> threshold);

[rows, cols] = find(harris_points);

figure,imshow(img) , hold on,
plot(cols, rows, 'ys'), title ('harris-coner-detector');


% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete. 20 random points
x = ceil(rand(20,1) * size(image,2));
y = ceil(rand(20,1) * size(image,1));

end

