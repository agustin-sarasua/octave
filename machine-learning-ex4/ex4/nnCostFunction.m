function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% FEEDFORWARD

% We know that we have 3 layers.
% a1 size 5000 x 401
a1 = [ones(m, 1) X];
% Theta1 size 25 x 401
z2 = a1 * Theta1';
% a2 size = 5000 x 26
m_a2 = size(z2, 1);
a2 = [ones(m_a2, 1) sigmoid(z2)];

% Theta2 size 10 x 26
z3 = a2 * Theta2';
% a3 size 5000 x 10
a3 = sigmoid(z3);
hx = a3;


% y vector 5000 x 1
I = eye(num_labels);
Y = zeros(m,num_labels);
for i=1:m
	Y(i, :)=I(y(i), :);
end;

% REGULATIZATION TERM

sum1 = sum(sum(Theta1(:, 2:size(Theta1, 2)).^2));
sum2 = sum(sum(Theta2(:, 2:size(Theta2, 2)).^2));
rt = (lambda/(2*m))*(sum1 + sum2);

% COST FUNCTION WITH REGULATIZATION TERM

J = (1/m) * sum(sum((-Y) .* log(hx) - ((1-Y) .* log(1 - hx)), 2)) + rt;

% BACKPROPAGATION to compute de GRADIENT so we can use an advanced optimizer such as fmincg

% delta(l)j are the error terms that measures how much a node was responsible for any error on our output.
% delta3 could be directly compute (hx - y) is the error associated with the output layer

% hx m x 10
% Y m x 10
delta3 = hx .- Y;

% delta3 m x 10
% Theta2 10 x 26
% z2 = m x 25
% Theta2_no_bias = 10 x 26
Theta2_no_bias = Theta2(:, 2:size(Theta2, 2));
delta2 = (delta3 * Theta2_no_bias) .* sigmoidGradient(z2);

% ACCUMULATE THE GRADIENTs

%a1_no_bias = a1(:, 2:size(a1, 2));
%a2_no_bias = a2(:, 2:size(a2, 2));


% delta2 		m x 25 
% delta3 		m x 10
% a1_no_bias 	m x 400
% a2_no_bias 	m x 25

D1 = (delta2'*a1);
D2 = (delta3'*a2);


Theta1_grad = (D1./m);
Theta2_grad = (D2./m);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
