function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
fprintf('Running gradientDecent with theta = %f\n', theta);
% Initialize some useful values
m = length(y); % number of training examples

% J_history will contain the cost of each of the iterations. Each iteration will have a diferent theta associated.
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % X mxn matrix
    % theta nx1 vector
    % h = hypothesis mx1 vector 
    h = X * theta;

    % y = mx1 vector of values
    % d = decrement
    d =  (alpha * (1/m) * (h - y)' * X);

    theta = theta - d';


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter));
end

end
