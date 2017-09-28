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
    
    % vector x with alL the population of the cities.
    % x0(i) is always 1 = X(:,1)
    % x1(i) value of feature 1 for the ith example. X(:,2)
    x = X(:,2);
    
    %h is a vector that contains the values of the hypothesis for each trainig example.
    h = theta(1) + (theta(2)*x);
    
    t0 = theta(1) - alpha * (1/m) * sum(h-y);
    t1  = theta(2) - alpha * (1/m) * sum((h - y) .* x);
    
    theta = [t0; t1];
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter));
end

end
