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
%



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

