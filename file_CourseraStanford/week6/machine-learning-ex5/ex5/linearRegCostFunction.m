function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% kich thuoc cua Theta bang so luong dac trung cua tap X
number_feature = size(theta);
% neu tap du lieu train chua co bias thi them bias vao
if size(X, 2)< number_feature
      temp_X = [ones(number_feature, 1) X];
else
      temp_X = X;
end
% tinh gia tri ham gia thuyet tuong ung voi gia tri X
h = temp_X*theta;
% tinh gia tri cua ham cost J
J = J+1/2/m*sum((h-y).^2)+lambda/2/m*sum(theta(2:end).^2);

% tinh gia tri cua gradient
grad = (1/m)*((temp_X'*(h-y)))+(lambda/m)*[0; theta(2:end)];




% =========================================================================

grad = grad(:);

end
