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

num_fea=size(theta);
[m1, n1]=size(X);
if m1<num_fea
    X_temp=[ones(m,1) X];
else
    X_temp=X;
end
% Compute vector of H function
h_theta=sigmoid(X_temp*theta);
h=-y.*log(h_theta)-(1-y).*log(1-h_theta);

J=1/m*sum(h)+lambda/2/m*sum(theta(2:num_fea).^2);

grad_temp=h_theta-y;
grad(1)=1/m*(grad_temp'*X(:, 1));
for i=2:num_fea
    grad(i)=1/m*(grad_temp'*X(:, i))+lambda/m*theta(i); 
end




% =============================================================

end
