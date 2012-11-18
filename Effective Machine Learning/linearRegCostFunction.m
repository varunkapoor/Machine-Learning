function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%	Compute the cost and gradient of regularized linear 
%   regression for a particular choice of theta.
%	set J to the cost and grad to the gradient.
%


J = (sum((X * theta - y) .^ 2) + lambda * sum(theta(2:end) .^ 2)) / (2 * m);

grad = (X' * (X*theta - y)) / m;
for i=2:n, 
    grad(i) += (lambda * theta(i)) / m;
end

grad = grad(:);

end
