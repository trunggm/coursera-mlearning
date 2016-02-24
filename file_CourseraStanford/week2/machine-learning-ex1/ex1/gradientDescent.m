function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
<<<<<<< HEAD

=======
X_temp = [ones(size(X)) X];
>>>>>>> ab8c585ef7a2f3b784b2e071fc66dbc64b37448d
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
<<<<<<< HEAD






=======
      % tinh gia tri ham gia thuyet h
      h = X_temp*theta';
      % tinh sai so
      error = h - y;
      % tinh gia tri theta
      theta = theta - alpha/m*(error'*X_temp);
>>>>>>> ab8c585ef7a2f3b784b2e071fc66dbc64b37448d

    % ============================================================

    % Save the cost J in every iteration    
<<<<<<< HEAD
    J_history(iter) = computeCost(X, y, theta);
=======
    J_history(iter) = 1/2/m*sum(error.^2);
>>>>>>> ab8c585ef7a2f3b784b2e071fc66dbc64b37448d

end

end
