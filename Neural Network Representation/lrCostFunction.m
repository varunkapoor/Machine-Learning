function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% 		the cost of a particular choice of theta.
%       set J to the cost.
%       compute the partial derivatives and set grad to the partial
%       derivatives of the cost w.r.t. each parameter in theta


J = (-1/m)*(y' * log(sigmoid(X*theta)) + (1-y)' * log(1-sigmoid(X*theta))) + (lambda/(2*m)) * (theta(2:end)' * theta(2:end));
grad = (1/m) * X' * (sigmoid(X*theta) - y);
grad(2:end) += (lambda/m) * theta(2:end);
grad = grad(:);

end
