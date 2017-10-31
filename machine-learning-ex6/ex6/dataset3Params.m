function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

t = [0.01 0.03 0.1 0.3 1 3 10 30];

res = zeros(length(t)^2,3);

count = 1;
for i = 1:length(t)
	C1 = t(i);
	for j = 1:length(t)
		sigma1 = t(j);
		model = svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1)); 	
		predictions = svmPredict(model, Xval);
		res(count,:) = [mean(double(predictions ~= yval)) C1 sigma1];
		count = count + 1;
	end;
end;

[m idx] = min(res(:,1))
C = res(idx, 2)
sigma = res(idx, 3)

% =========================================================================

end
