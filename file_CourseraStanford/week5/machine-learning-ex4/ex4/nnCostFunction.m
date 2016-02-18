function [J, grad] = nnCostFunction(nn_params, ...
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
% fix lai vector y
Y = zeros(m, num_labels);
for i=1:m
      Y(i, mod(y(i),10))=1;
end
% them dac trung bias vao X
X = [ones(m, 1), X];
% tinh gia tri cua ham gia thuyet H
a_2 = sigmoid(X*Theta1');
% them dac trung bias vao a_1
a_2 = [ones(m, 1), a_2];
% tinh gia tri cua ham gia thuyet H
h = sigmoid(a_2*Theta2');
% tinh gia tri ham cost J
J = J + 1/m*sum(sum((-Y.*log(h)-(1-Y).*log(1-h)),2))+...
      lambda/2/m*(sum(sum(Theta1.^2))+sum(sum(Theta2.^2)));
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
% khoi tao gia tri cua  deltaC
deltaC1 = zeros(input_layer_size, 1); 
deltaC2 = zeros(hidden_layer_size, 1); 
for i=1:m
% thuc hien lan truyen thuan
      % gan a1=du lieu train
      a1 = X(i,:)';
      % tinh gia tri active a2
      z2 = sigmoid(Theta1*a1);
      % them gia tri 1 vao tinh toan
      a2 = [1; z2];
      % tinh gia tri cua ham gia thuyet tuong ung voi gia tri cua Xi
      a3 = sigmoid(Theta2*a2);
      h = a3; % gia tri ham gia thuyet h = a3
% thuc hien lan truyen nguoc
      delta3=a3-Y(i,:)'; %sai so cua cac nut o output layer(la vector hang)
      % tinh sai so o ca layer tiep theo
      delta2=(Theta2'*a3).*(a2.*(1-a2)); % tinh gia tri sai so 
                                          % cua cac nut o hiden layer
      % cap nhat gia tri cua cac deltaC
      delta2=delta2(2:end);
      
end
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
