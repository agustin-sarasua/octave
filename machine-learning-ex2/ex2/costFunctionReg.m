function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% X is a matrix of 118x28 (118 training examples of 28 features, colulmn0 = 1) 
% theta is a column vector of 28x1 
% X * theta = column vector of 118x1 => hypothesis is a column vector of 118x1
% y is a column vector of 118x1 full of 0s and 1s
% -y' * log(hypothesis) = 1x118 * 118x1 = 1x1
% (1 - y)'*log(1-hypothesis) = 1x118 * 118x1 = 1x1
% (lambda/(2*m)) = Real number
% theta_reg'*theta_reg = 1x28 * 28x1 = 1x1


hypothesis = sigmoid(X * theta);
% this is because we do not want to regularize theta0 (which is equal to theta(1))
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

% A vectorize implementation

J = (1/m)*(-y'* log(hypothesis) - (1 - y)'*log(1-hypothesis))+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(hypothesis-y)+lambda*theta_reg);

% =============================================================

end
